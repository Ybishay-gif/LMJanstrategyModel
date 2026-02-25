#!/usr/bin/env python3
"""Verify recommendation strategy changes impact selected price points and bind projections.

Run:
  .venv/bin/python scripts/strategy_mode_impact_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import asdict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_PATHS, Settings, OPTIMIZATION_MODES
from src import app


def settings_for_mode(mode: str) -> Settings:
    return Settings(
        max_cpc_increase_pct=45,
        min_bids_channel_state=5,
        cpc_penalty_weight=0.65,
        growth_weight=0.70,
        profit_weight=0.30,
        aggressive_cutoff=0.40,
        controlled_cutoff=0.25,
        maintain_cutoff=0.10,
        min_intent_for_scale=0.65,
        roe_pullback_floor=-0.45,
        cr_pullback_ceiling=1.35,
        max_adj_strongest=45,
        max_adj_moderate=35,
        max_adj_minimal=25,
        max_adj_constrained=15,
        min_clicks_intent_sig=80,
        min_bids_price_sig=75,
        min_clicks_price_sig=30,
        min_binds_perf_sig=8,
        optimization_mode=mode,
        max_perf_drop=0.15,
        min_new_performance=0.80,
    )


def build_for_mode(mode: str) -> pd.DataFrame:
    p = DEFAULT_PATHS
    mt = (
        app.file_mtime(p["state_strategy"]),
        app.file_mtime(p["state_data"]),
        app.file_mtime(p["state_seg"]),
        app.file_mtime(p["channel_group"]),
        app.file_mtime(p["channel_price_exp"]),
        app.file_mtime(p["channel_state"]),
    )
    rec_df, *_ = app.build_all_from_paths(
        p["state_strategy"],
        p["state_data"],
        p["state_seg"],
        p["channel_group"],
        p["channel_price_exp"],
        p["channel_state"],
        settings_for_mode(mode),
        mt,
    )
    return rec_df


def summarize(rec_df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(rec_df)),
        "avg_suggested_adj": float(rec_df["Suggested Price Adjustment %"].fillna(0).mean()),
        "sum_additional_clicks": float(rec_df["Expected Additional Clicks"].fillna(0).sum()),
        "sum_additional_binds": float(rec_df["Expected Additional Binds"].fillna(0).sum()),
        "sum_additional_cost": float(rec_df["Expected Additional Cost"].fillna(0).sum()),
        "non_zero_suggested_rows": int((rec_df["Suggested Price Adjustment %"].fillna(0) != 0).sum()),
    }


def changed_rows(a: pd.DataFrame, b: pd.DataFrame) -> int:
    keys = ["State", "Channel Groups", "Segment"]
    aa = a[keys + ["Suggested Price Adjustment %"]].copy()
    bb = b[keys + ["Suggested Price Adjustment %"]].copy()
    merged = aa.merge(bb, on=keys, how="inner", suffixes=("_a", "_b"))
    delta = (
        pd.to_numeric(merged["Suggested Price Adjustment %_a"], errors="coerce").fillna(0)
        - pd.to_numeric(merged["Suggested Price Adjustment %_b"], errors="coerce").fillna(0)
    ).abs()
    return int((delta > 1e-9).sum())


def main() -> int:
    by_mode_df: dict[str, pd.DataFrame] = {mode: build_for_mode(mode) for mode in OPTIMIZATION_MODES}
    by_mode_summary = {mode: summarize(df) for mode, df in by_mode_df.items()}

    max_growth = by_mode_df["Max Growth"]
    optimize_cost = by_mode_df["Optimize Cost"]
    balanced = by_mode_df["Balanced"]

    growth_vs_cost_changed = changed_rows(max_growth, optimize_cost)
    growth_vs_balanced_changed = changed_rows(max_growth, balanced)

    assert growth_vs_cost_changed > 0, "No row-level recommendation change between Max Growth and Optimize Cost"
    assert growth_vs_balanced_changed > 0, "No row-level recommendation change between Max Growth and Balanced"
    assert (
        by_mode_summary["Max Growth"]["sum_additional_binds"]
        > by_mode_summary["Optimize Cost"]["sum_additional_binds"]
    ), "Max Growth should produce higher binds than Optimize Cost"

    result = {
        "status": "PASS",
        "changed_rows": {
            "max_growth_vs_optimize_cost": growth_vs_cost_changed,
            "max_growth_vs_balanced": growth_vs_balanced_changed,
        },
        "by_mode_summary": by_mode_summary,
        "settings_reference": asdict(settings_for_mode("Balanced")),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

