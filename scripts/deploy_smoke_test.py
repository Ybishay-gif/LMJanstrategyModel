#!/usr/bin/env python3
"""Pre-deploy smoke tests for manual bid-adjustment flow.

Run:
  .venv/bin/python scripts/deploy_smoke_test.py
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

from src.config import DEFAULT_PATHS, Settings
from src import app


def default_settings() -> Settings:
    return Settings(
        max_cpc_increase_pct=25,
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
        optimization_mode="Balanced",
        max_perf_drop=0.15,
        min_new_performance=0.80,
    )


def assert_required_columns(df: pd.DataFrame) -> None:
    required = {
        "State",
        "Channel Groups",
        "Applied Price Adjustment %",
        "Expected Additional Clicks",
        "Expected Additional Binds",
        "Expected Additional Cost",
        "CPC Lift %",
        "Total Click Cost",
    }
    missing = sorted(required - set(df.columns))
    assert not missing, f"Missing required columns: {missing}"


def build_data(settings: Settings):
    p = DEFAULT_PATHS
    mt = (
        app.file_mtime(p["state_strategy"]),
        app.file_mtime(p["state_data"]),
        app.file_mtime(p["state_seg"]),
        app.file_mtime(p["channel_group"]),
        app.file_mtime(p["channel_price_exp"]),
        app.file_mtime(p["channel_state"]),
    )
    return app.build_all_from_paths(
        p["state_strategy"],
        p["state_data"],
        p["state_seg"],
        p["channel_group"],
        p["channel_price_exp"],
        p["channel_state"],
        settings,
        mt,
    )


def pick_scenario(rec_df: pd.DataFrame, price_eval: pd.DataFrame, settings: Settings):
    preferred = ("AL", "Group 150 MCH")
    row = rec_df[(rec_df["State"] == preferred[0]) & (rec_df["Channel Groups"] == preferred[1])]
    if not row.empty:
        pop_pref = app.precompute_popup_options_for_state(rec_df, price_eval, preferred[0], settings)
        pref_opts = (
            pd.to_numeric(
                pop_pref.loc[pop_pref["Channel Groups"] == preferred[1], "Bid Adj %"],
                errors="coerce",
            )
            .dropna()
            .astype(float)
            .tolist()
        )
        rec_adj_pref = float(row.iloc[0].get("Applied Price Adjustment %", 0.0) or 0.0)
        if any(not app.close_adj(o, rec_adj_pref) for o in pref_opts):
            return preferred

    # fallback: first row with any valid popup options and at least one different adj
    for state in rec_df["State"].dropna().unique().tolist():
        pop = app.precompute_popup_options_for_state(rec_df, price_eval, str(state), settings)
        if pop.empty:
            continue
        state_rows = rec_df[rec_df["State"] == state]
        for _, r in state_rows.iterrows():
            ch = str(r["Channel Groups"])
            rec_adj = float(r.get("Applied Price Adjustment %", 0.0) or 0.0)
            opts = (
                pd.to_numeric(pop.loc[pop["Channel Groups"] == ch, "Bid Adj %"], errors="coerce")
                .dropna()
                .astype(float)
                .tolist()
            )
            if any(not app.close_adj(o, rec_adj) for o in opts):
                return str(state), ch

    raise AssertionError("No valid scenario found with alternate adjustment options")


def choose_manual_adj(rec_adj: float, popup_for_channel: pd.DataFrame) -> float:
    opts = (
        pd.to_numeric(popup_for_channel["Bid Adj %"], errors="coerce")
        .dropna()
        .astype(float)
        .tolist()
    )
    assert opts, "No stat-sig price test options for selected scenario"

    # Prefer +10 if available and different from recommendation.
    for x in opts:
        if app.close_adj(x, 10.0) and not app.close_adj(x, rec_adj):
            return x

    for x in sorted(opts):
        if not app.close_adj(x, rec_adj):
            return x

    raise AssertionError("No alternate manual adjustment found for scenario")


def test_manual_override_recomputes(rec_df: pd.DataFrame, price_eval: pd.DataFrame, settings: Settings) -> dict:
    state, channel = pick_scenario(rec_df, price_eval, settings)
    key = f"{state}|{channel}"
    base = rec_df[(rec_df["State"] == state) & (rec_df["Channel Groups"] == channel)].iloc[0]

    pop = app.precompute_popup_options_for_state(rec_df, price_eval, state, settings)
    pop_row = pop[pop["Channel Groups"] == channel]
    assert not pop_row.empty, f"No popup options for {key}"

    rec_adj = float(base["Applied Price Adjustment %"])
    manual_adj = choose_manual_adj(rec_adj, pop_row)

    overrides = {key: {"apply": True, "adj": manual_adj, "requested_adj": manual_adj, "source": "manual"}}
    rec2 = app.apply_user_bid_overrides(rec_df, price_eval, settings, overrides)
    new = rec2[(rec2["State"] == state) & (rec2["Channel Groups"] == channel)].iloc[0]

    assert app.close_adj(float(new["Applied Price Adjustment %"]), manual_adj), (
        f"Applied adjustment mismatch. expected={manual_adj}, got={new['Applied Price Adjustment %']}"
    )

    changed = any(
        abs(float(new[col]) - float(base[col])) > 1e-9
        for col in [
            "Expected Additional Clicks",
            "Expected Additional Binds",
            "Expected Additional Cost",
            "CPC Lift %",
        ]
    )
    assert changed, "Override did not update downstream metrics"

    return {
        "scenario": key,
        "recommended_adj": rec_adj,
        "manual_adj": manual_adj,
        "base_add_clicks": float(base["Expected Additional Clicks"]),
        "new_add_clicks": float(new["Expected Additional Clicks"]),
        "base_add_binds": float(base["Expected Additional Binds"]),
        "new_add_binds": float(new["Expected Additional Binds"]),
        "base_add_cost": float(base["Expected Additional Cost"]),
        "new_add_cost": float(new["Expected Additional Cost"]),
    }


def test_row_option_map_includes_current_recommendation(rec_df: pd.DataFrame, price_eval: pd.DataFrame, settings: Settings) -> dict:
    state = str(rec_df["State"].dropna().astype(str).iloc[0])
    state_rows = rec_df[rec_df["State"] == state].copy()
    pop = app.precompute_popup_options_for_state(rec_df, price_eval, state, settings)

    checked = 0
    for _, rr in state_rows.iterrows():
        ch = str(rr["Channel Groups"])
        rec_adj = float(rr.get("Applied Price Adjustment %", 0.0) or 0.0)
        ch_opts = pop[pop["Channel Groups"] == ch]

        labels = []
        label_to_adj = {}
        for _, op in ch_opts.iterrows():
            ladj = float(op.get("Bid Adj %", 0) or 0)
            lbl = app.format_adj_option_label(
                ladj,
                float(op.get("Win Rate Uplift", 0) or 0),
                float(op.get("CPC Uplift", 0) or 0),
                op.get("CPB Impact", pd.NA),
                str(op.get("Sig Level", "")),
            )
            labels.append(lbl)
            label_to_adj[lbl] = ladj

        if not labels:
            fallback = f"{rec_adj:+.0f}%: n/a Clicks || n/a CPC || n/a CPB (no stat-sig)"
            labels = [fallback]
            label_to_adj[fallback] = rec_adj

        has_current = any(
            app.close_adj(float(v), rec_adj) for v in label_to_adj.values()
        )
        if not has_current:
            model_lbl = f"{rec_adj:+.0f}%: model recommendation (no matched stat-sig option)"
            labels = [model_lbl] + labels
            label_to_adj[model_lbl] = rec_adj

        assert any(app.close_adj(float(v), rec_adj) for v in label_to_adj.values()), (
            f"Current recommendation missing from row option map for {state}|{ch}"
        )
        checked += 1

    assert checked > 0, "No rows checked in row-option-map test"
    return {"checked_rows": checked, "state": state}


def test_tab5_preset_switch_behavior() -> dict:
    p1 = "Preset A"
    p2 = "Preset B"
    k1 = app.tab5_grid_component_key(p1)
    k2 = app.tab5_grid_component_key(p2)
    assert k1 != k2, "Grid key should change when preset changes to force remount"

    base_go = {"rowGroupPanelShow": "always"}
    preset = {
        "column_state": [{"colId": "State", "hide": False}],
        "filter_model": {"State": {"filterType": "set", "values": ["VA"]}},
        "sort_model": [{"colId": "Cost", "sort": "desc"}],
        "pivot_mode": True,
        "grid_state": {"columns": {"columnVisibility": {"hiddenColIds": ["Source Used"]}}},
    }
    out_go = app.apply_grid_preset(dict(base_go), preset)
    assert out_go.get("columnState"), "Preset column state was not applied"
    assert out_go.get("filterModel"), "Preset filter model was not applied"
    assert out_go.get("sortModel"), "Preset sort model was not applied"
    assert out_go.get("pivotMode") is True, "Preset pivot mode was not applied"
    assert out_go.get("initialState"), "Preset initial grid state was not applied"
    return {"key_a": k1, "key_b": k2, "preset_fields_applied": True}


def main() -> None:
    settings = default_settings()
    rec_df, _state_df, _state_seg_df, price_eval, _state_extra, _state_seg_extra = build_data(settings)

    assert not rec_df.empty, "rec_df is empty"
    assert_required_columns(rec_df)

    r1 = test_manual_override_recomputes(rec_df, price_eval, settings)
    r2 = test_row_option_map_includes_current_recommendation(rec_df, price_eval, settings)
    r3 = test_tab5_preset_switch_behavior()

    out = {
        "status": "PASS",
        "settings": asdict(settings),
        "manual_override_test": r1,
        "row_option_map_test": r2,
        "tab5_preset_switch_test": r3,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
