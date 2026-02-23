import re
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

st.set_page_config(page_title="Insurance Growth Navigator", layout="wide")

DEFAULT_PATHS = {
    "state_strategy": "data/state_strategy.txt",
    "state_data": "data/state_data.csv",
    "state_seg": "data/state_seg_data.csv",
    "channel_group": "data/channel_group_data.csv",
    "channel_price_exp": "data/channel_price_exploration.csv",
    "channel_state": "data/channel_group_state.csv",
}

STRATEGY_SCALE = {
    "Strongest Momentum": 1.00,
    "Moderate Momentum": 0.80,
    "Minimal Growth": 0.55,
    "LTV Constrained": 0.35,
    "Closure Constrained": 0.30,
    "Inactive/Low Spend": 0.10,
}

STRATEGY_COLOR = {
    "Strongest Momentum": "#FFD400",
    "Moderate Momentum": "#74D2D4",
    "LTV Constrained": "#B2ACE6",
    "Closure Constrained": "#1B185A",
    "Minimal Growth": "#B5B5B5",
    "Inactive/Low Spend": "#F3F4F6",
}

PERFORMANCE_GROUP_COLOR = {
    "Top Performance": "#16A34A",
    "Strong Performance": "#4ADE80",
    "Balanced": "#A3E635",
    "Mixed Risk": "#FACC15",
    "Weak Performance": "#F97316",
    "Poor Performance": "#DC2626",
    "Low Sig - Review": "#94A3B8",
}

CONFLICT_PERF_COLOR = {
    "Good | Full Match": "#166534",
    "Good | Small Conflict": "#22C55E",
    "Good | High Conflict": "#86EFAC",
    "OK | Full Match": "#A16207",
    "OK | Small Conflict": "#EAB308",
    "OK | High Conflict": "#FDE047",
    "Poor | Full Match": "#991B1B",
    "Poor | Small Conflict": "#EF4444",
    "Poor | High Conflict": "#FCA5A5",
    "Unknown | Unknown": "#94A3B8",
}

STATE_CENTER = {
    "AL": (32.7, -86.7), "AK": (64.2, -149.5), "AZ": (34.3, -111.7), "AR": (34.9, -92.4),
    "CA": (37.3, -119.7), "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5),
    "FL": (27.8, -81.7), "GA": (32.7, -83.3), "HI": (20.8, -157.5), "ID": (44.1, -114.7),
    "IL": (40.0, -89.2), "IN": (40.0, -86.1), "IA": (42.1, -93.5), "KS": (38.5, -98.0),
    "KY": (37.5, -85.3), "LA": (31.2, -92.3), "ME": (45.2, -69.0), "MD": (39.0, -76.7),
    "MA": (42.3, -71.8), "MI": (44.3, -85.6), "MN": (46.4, -94.6), "MS": (32.7, -89.7),
    "MO": (38.6, -92.6), "MT": (46.9, -110.0), "NE": (41.5, -99.8), "NV": (38.8, -116.4),
    "NH": (43.7, -71.6), "NJ": (40.1, -74.7), "NM": (34.5, -106.1), "NY": (42.9, -75.0),
    "NC": (35.5, -79.4), "ND": (47.5, -100.5), "OH": (40.3, -82.8), "OK": (35.6, -97.5),
    "OR": (44.0, -120.5), "PA": (41.0, -77.5), "RI": (41.7, -71.6), "SC": (33.8, -80.9),
    "SD": (44.5, -100.2), "TN": (35.8, -86.4), "TX": (31.5, -99.3), "UT": (39.3, -111.7),
    "VT": (44.1, -72.7), "VA": (37.5, -78.7), "WA": (47.4, -120.7), "WV": (38.6, -80.6),
    "WI": (44.5, -89.8), "WY": (43.0, -107.6), "DC": (38.9, -77.0),
}

DARK_CSS = """
<style>
:root {
  --bg: #0b1220;
  --panel: #121a2b;
  --panel-soft: #18233a;
  --text: #e6edf9;
  --muted: #97a6c2;
  --accent: #3dd6ff;
  --good: #4ade80;
}
.stApp {
  background: radial-gradient(circle at 20% 0%, #1a2440 0%, #0b1220 48%, #080d18 100%);
  color: var(--text);
}
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #0a1020 100%);
}
.hero-card {
  padding: 14px 16px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(61,214,255,0.18), rgba(96,165,250,0.10));
  border: 1px solid rgba(61,214,255,0.3);
  margin-bottom: 12px;
}
.tiny-note {
  color: var(--muted);
  font-size: 0.90rem;
}
.conflict-legend { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-top:8px; }
.conflict-item { display:inline-flex; align-items:center; gap:8px; color:var(--text); font-size:0.85rem; }
.swatch { width:26px; height:16px; border:1px solid rgba(148,163,184,0.6); border-radius:4px; }
.striped {
  background: repeating-linear-gradient(
    45deg,
    #f3f4f6 0px,
    #f3f4f6 12px,
    #e5e7eb 12px,
    #e5e7eb 24px
  );
}
.dots {
  background-color: #ffffff;
  background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
  background-size: 12px 12px;
  background-position: 0 0;
}
.solid { background: #86efac; }
.alt-card {
  border: 1px solid rgba(148,163,184,0.35);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(15,23,42,0.55);
  min-height: 170px;
}
.alt-card h5 {
  margin: 0 0 8px 0;
  font-size: 1rem;
}
.alt-kpi {
  font-size: 0.82rem;
  color: #cbd5e1;
  margin: 2px 0;
}
</style>
"""

LIGHT_CSS = """
<style>
.hero-card {
  padding: 14px 16px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(2,132,199,0.12), rgba(14,165,233,0.08));
  border: 1px solid rgba(2,132,199,0.25);
  margin-bottom: 12px;
}
.tiny-note {
  color: #334155;
  font-size: 0.90rem;
}
.conflict-legend { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-top:8px; }
.conflict-item { display:inline-flex; align-items:center; gap:8px; color:#0f172a; font-size:0.85rem; }
.swatch { width:26px; height:16px; border:1px solid #94a3b8; border-radius:4px; }
.striped {
  background: repeating-linear-gradient(
    45deg,
    #f3f4f6 0px,
    #f3f4f6 12px,
    #e5e7eb 12px,
    #e5e7eb 24px
  );
}
.dots {
  background-color: #ffffff;
  background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
  background-size: 12px 12px;
  background-position: 0 0;
}
.solid { background: #86efac; }
.alt-card {
  border: 1px solid rgba(148,163,184,0.35);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(248,250,252,0.9);
  min-height: 170px;
}
.alt-card h5 {
  margin: 0 0 8px 0;
  font-size: 1rem;
}
.alt-kpi {
  font-size: 0.82rem;
  color: #334155;
  margin: 2px 0;
}
</style>
"""


@dataclass
class Settings:
    max_cpc_increase_pct: float
    min_bids_channel_state: int
    cpc_penalty_weight: float
    growth_weight: float
    profit_weight: float
    aggressive_cutoff: float
    controlled_cutoff: float
    maintain_cutoff: float
    min_intent_for_scale: float
    roe_pullback_floor: float
    cr_pullback_ceiling: float
    max_adj_strongest: float
    max_adj_moderate: float
    max_adj_minimal: float
    max_adj_constrained: float
    min_clicks_intent_sig: int
    min_bids_price_sig: int
    min_clicks_price_sig: int
    min_binds_perf_sig: int
    optimization_mode: str
    max_perf_drop: float
    min_new_performance: float


OPTIMIZATION_MODES = [
    "Max Growth",
    "Growth Leaning",
    "Balanced",
    "Cost Leaning",
    "Optimize Cost",
]


def mode_factor(mode: str) -> float:
    m = {
        "Max Growth": 1.0,
        "Growth Leaning": 0.75,
        "Balanced": 0.5,
        "Cost Leaning": 0.25,
        "Optimize Cost": 0.0,
    }
    return m.get(mode, 0.5)


def effective_cpc_cap_pct(settings: Settings) -> float:
    # Cost mode keeps a tight cap; growth mode allows wider CPC expansion,
    # but never above the explicit guardrail slider.
    f = mode_factor(settings.optimization_mode)
    mode_cap = 12.0 + (45.0 - 12.0) * f
    return min(float(settings.max_cpc_increase_pct), mode_cap)


def effective_cpc_penalty(settings: Settings) -> float:
    # Cost mode penalizes CPC harder; growth mode softens the penalty.
    f = mode_factor(settings.optimization_mode)
    mult = 1.6 - 1.25 * f  # 1.6 -> 0.35
    return settings.cpc_penalty_weight * mult


def classify_perf_group(roe: float, combined_ratio: float, performance: float, binds: float, min_binds_sig: int) -> tuple[str, bool]:
    sig = pd.notna(binds) and (binds >= min_binds_sig)
    if not sig:
        return "Low Sig - Review", False

    strong = 0
    weak = 0

    if pd.notna(roe):
        if roe > 0:
            strong += 1
        elif roe < -0.25:
            weak += 1
    if pd.notna(combined_ratio):
        if combined_ratio < 1.00:
            strong += 1
        elif combined_ratio > 1.15:
            weak += 1
    if pd.notna(performance):
        if performance >= 1.00:
            strong += 1
        elif performance < 0.80:
            weak += 1

    if strong == 3:
        return "Top Performance", True
    if weak == 3:
        return "Poor Performance", True
    if strong >= 2 and weak == 0:
        return "Strong Performance", True
    if weak >= 2 and strong == 0:
        return "Weak Performance", True
    if weak >= 1 and strong <= 1:
        return "Mixed Risk", True
    return "Balanced", True


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_state_strategy(path: str) -> pd.DataFrame:
    raw = Path(path).read_text(errors="ignore")
    return parse_state_strategy_text(raw)


def parse_state_strategy_text(raw: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    if len(lines) >= 2 and lines[0].lower() == "state":
        lines = lines[2:]

    pairs: list[tuple[str, str]] = []
    for i in range(0, len(lines) - 1, 2):
        state = lines[i].upper()
        bucket = lines[i + 1]
        if re.fullmatch(r"[A-Z]{2}", state):
            pairs.append((state, bucket))

    return pd.DataFrame(pairs, columns=["State", "Strategy Bucket"]).drop_duplicates("State")


def to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    pct_mask = s.str.contains("%")
    out = pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")
    out[pct_mask] = out[pct_mask] / 100.0
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out


def extract_segment(channel_group_name: str) -> str:
    m = re.search(r"\b(MCH|MCR|SCH|SCR)\b", str(channel_group_name))
    return m.group(1) if m else "UNKNOWN"


def quantile_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    uniq = series.nunique(dropna=True)
    if uniq < 2:
        return pd.Series([labels[0]] * len(series), index=series.index)
    q = min(len(labels), uniq)
    out = pd.qcut(series.rank(method="first"), q=q, labels=labels[:q], duplicates="drop")
    return out.astype(str)


def strategy_max_adjustment(bucket: str, settings: Settings) -> float:
    if bucket == "Strongest Momentum":
        return settings.max_adj_strongest
    if bucket == "Moderate Momentum":
        return settings.max_adj_moderate
    if bucket == "Minimal Growth":
        return settings.max_adj_minimal
    if bucket in {"LTV Constrained", "Closure Constrained", "Inactive/Low Spend"}:
        return settings.max_adj_constrained
    return settings.max_adj_minimal


def _build_effect_dicts(price_eval_df: pd.DataFrame, settings: Settings) -> tuple[dict, dict]:
    px = price_eval_df.copy()
    if "Stat Sig Price Point" in px.columns:
        px = px[px["Stat Sig Price Point"] == True]
    if "CPC Lift %" in px.columns:
        px = px[px["CPC Lift %"].fillna(0) <= effective_cpc_cap_pct(settings) / 100.0]

    state_dict: dict[tuple[str, str], pd.DataFrame] = {}
    if "State" in px.columns:
        st_eff = (
            px[["State", "Channel Groups", "Price Adjustment Percent", "Clicks Lift %", "Win Rate Lift %", "CPC Lift %"]]
            .groupby(["State", "Channel Groups", "Price Adjustment Percent"], as_index=False)
            .mean(numeric_only=True)
        )
        state_dict = {
            (str(st), str(ch)): g.sort_values("Price Adjustment Percent")
            for (st, ch), g in st_eff.groupby(["State", "Channel Groups"])
        }

    ch_eff = (
        px[["Channel Groups", "Price Adjustment Percent", "Clicks Lift %", "Win Rate Lift %", "CPC Lift %"]]
        .groupby(["Channel Groups", "Price Adjustment Percent"], as_index=False)
        .mean(numeric_only=True)
    )
    channel_dict: dict[str, pd.DataFrame] = {
        str(ch): g.sort_values("Price Adjustment Percent") for ch, g in ch_eff.groupby("Channel Groups")
    }
    return state_dict, channel_dict


def _assign_best_price_points(rec: pd.DataFrame, price_eval_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    px = price_eval_df.copy()
    if "Stat Sig Price Point" in px.columns:
        px = px[px["Stat Sig Price Point"] == True]
    if "CPC Lift %" in px.columns:
        px = px[px["CPC Lift %"].fillna(0) <= effective_cpc_cap_pct(settings) / 100.0]
    px = px[(px["Price Adjustment Percent"].fillna(0) >= 0) & (px["Win Rate Lift %"].fillna(0) >= 0)]

    state_curves: dict[tuple[str, str], pd.DataFrame] = {}
    if "State" in px.columns:
        state_curves = {
            (str(st), str(ch)): g.sort_values("Price Adjustment Percent")
            for (st, ch), g in px.groupby(["State", "Channel Groups"])
        }
    ch_curves = {str(ch): g.sort_values("Price Adjustment Percent") for ch, g in px.groupby("Channel Groups")}

    def _pick_from_curve(curve: pd.DataFrame, row: pd.Series) -> pd.Series | None:
        if curve is None or curve.empty:
            return None
        bids = float(row.get("Bids", 0) or 0)
        clicks = float(row.get("Clicks", 0) or 0)
        avg_cpc = float(row.get("Avg. CPC", 0) or 0)
        binds = float(row.get("Binds", 0) or 0)
        current_cost = float(row.get("Total Click Cost", np.nan))
        if pd.isna(current_cost):
            current_cost = clicks * avg_cpc

        base_wr = float(row.get("Bids to Clicks", np.nan))
        if pd.isna(base_wr):
            base_wr = (clicks / bids) if bids > 0 else 0.0
        c2b_seg = float(row.get("Seg Clicks to Binds", np.nan))
        c2b_ch = float(row.get("Clicks to Binds", np.nan))
        c2b = c2b_seg if (bids >= settings.min_bids_channel_state and not pd.isna(c2b_seg)) else c2b_ch
        if pd.isna(c2b):
            c2b = 0.0

        target_cpb = float(row.get("Target CPB", np.nan))
        actual_cpb = float(row.get("CPB", np.nan))
        if pd.isna(actual_cpb) and binds > 0:
            actual_cpb = current_cost / binds
        actual_perf = np.nan
        if pd.notna(target_cpb) and target_cpb > 0 and pd.notna(actual_cpb) and actual_cpb > 0:
            actual_perf = target_cpb / actual_cpb

        scored = []
        for _, c in curve.iterrows():
            wr_lift = float(c.get("Win Rate Lift %", 0) or 0)
            cpc_lift = float(c.get("CPC Lift %", 0) or 0)
            add_clicks = bids * base_wr * max(wr_lift, 0.0)
            add_binds = add_clicks * c2b
            new_binds = binds + add_binds
            new_cost = (clicks + add_clicks) * avg_cpc * (1 + cpc_lift)
            new_cpb = (new_cost / new_binds) if new_binds > 0 else np.nan
            new_perf = np.nan
            if pd.notna(target_cpb) and target_cpb > 0 and pd.notna(new_cpb) and new_cpb > 0:
                new_perf = target_cpb / new_cpb
            perf_drop = 0.0
            if pd.notna(actual_perf) and pd.notna(new_perf):
                perf_drop = max(actual_perf - new_perf, 0.0)
            perf_ok = True
            if pd.notna(new_perf):
                perf_ok = (new_perf >= settings.min_new_performance) and (perf_drop <= settings.max_perf_drop)
            scored.append(
                {
                    "cand": c,
                    "add_binds": add_binds,
                    "add_clicks": add_clicks,
                    "perf_drop": perf_drop,
                    "new_perf": new_perf,
                    "perf_ok": perf_ok,
                }
            )
        if not scored:
            return None
        valid = [x for x in scored if x["perf_ok"]]
        if valid:
            best = sorted(
                valid,
                key=lambda x: (x["add_binds"], -float(x["cand"].get("CPC Lift %", 0) or 0), -float(x["cand"].get("Price Adjustment Percent", 0) or 0)),
                reverse=True,
            )[0]
            return best["cand"]
        # If all points hurt performance too much, choose the most conservative feasible degradation.
        best = sorted(
            scored,
            key=lambda x: (x["perf_drop"], -x["add_binds"], float(x["cand"].get("Price Adjustment Percent", 0) or 0)),
        )[0]
        return best["cand"]

    def _lookup(row: pd.Series) -> pd.Series:
        key_sc = (str(row.get("State", "")), str(row.get("Channel Groups", "")))
        srow = _pick_from_curve(state_curves.get(key_sc), row)
        if srow is not None:
            return pd.Series(
                [
                    srow.get("Price Adjustment Percent", 0.0),
                    srow.get("Growth Opportunity Score", 0.0),
                    srow.get("Clicks Lift %", 0.0),
                    srow.get("CPC Lift %", 0.0),
                    srow.get("Win Rate Lift %", 0.0),
                    True,
                ]
            )
        crow = _pick_from_curve(ch_curves.get(str(row.get("Channel Groups", ""))), row)
        if crow is not None:
            return pd.Series(
                [
                    crow.get("Price Adjustment Percent", 0.0),
                    crow.get("Growth Opportunity Score", 0.0),
                    crow.get("Clicks Lift %", 0.0),
                    crow.get("CPC Lift %", 0.0),
                    crow.get("Win Rate Lift %", 0.0),
                    True,
                ]
            )
        return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, False])

    out = rec.copy()
    mapped = out.apply(_lookup, axis=1)
    mapped.columns = [
        "Price Adjustment Percent",
        "Growth Opportunity Score",
        "Clicks Lift %",
        "CPC Lift %",
        "Win Rate Lift %",
        "Has Sig Price Evidence",
    ]
    out[[
        "Price Adjustment Percent",
        "Growth Opportunity Score",
        "Clicks Lift %",
        "CPC Lift %",
        "Win Rate Lift %",
        "Has Sig Price Evidence",
    ]] = mapped
    return out


def apply_price_effects(
    rec: pd.DataFrame, price_eval_df: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    state_dict, effect_dict = _build_effect_dicts(price_eval_df, settings)

    def lookup(row: pd.Series) -> pd.Series:
        g = state_dict.get((str(row.get("State", "")), str(row["Channel Groups"])))
        if g is None or g.empty:
            g = effect_dict.get(str(row["Channel Groups"]))
        if g is None or g.empty:
            return pd.Series([0.0, 0.0, 0.0, 0.0])
        target = row["Suggested Price Adjustment %"]
        g2 = g.copy()
        g2["dist"] = (g2["Price Adjustment Percent"] - target).abs()
        near = g2.sort_values(["dist", "Price Adjustment Percent"], ascending=[True, True]).iloc[0]
        return pd.Series(
            [
                near["Price Adjustment Percent"],
                near["Clicks Lift %"],
                near["Win Rate Lift %"],
                near["CPC Lift %"],
            ]
        )

    mapped = rec.apply(lookup, axis=1)
    mapped.columns = ["Applied Price Adjustment %", "Clicks Lift %", "Win Rate Lift %", "CPC Lift %"]
    out = rec.copy()
    out[["Applied Price Adjustment %", "Clicks Lift %", "Win Rate Lift %", "CPC Lift %"]] = mapped
    return out


@st.cache_data(show_spinner=False, hash_funcs={Settings: lambda s: tuple(vars(s).items())})
def prepare_state(state_df: pd.DataFrame, strategy_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    df = normalize_columns(state_df)
    for col in ["ROE", "Combined Ratio", "Performance", "CPB", "Target CPB", "Clicks", "Binds", "Avg. MRLTV", "Quotes to Binds"]:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    out = df.merge(strategy_df, on="State", how="left")
    out["Strategy Scale"] = out["Strategy Bucket"].map(STRATEGY_SCALE).fillna(0.5)

    out["Profitability Score"] = (
        0.35 * out["Performance"].fillna(0)
        + 0.30 * out["ROE"].fillna(-0.2)
        + 0.20 * (1 - out["Combined Ratio"].fillna(1.15))
        + 0.15 * ((out["Avg. MRLTV"].fillna(out["Avg. MRLTV"].median()) / out["Avg. MRLTV"].median()) - 1)
    )

    out["ROE Band"] = np.select(
        [out["ROE"] >= 0, out["ROE"].between(-0.20, 0, inclusive="left")],
        ["Good", "OK"],
        default="Poor",
    )
    out["CR Band"] = np.select(
        [out["Combined Ratio"] < 1.00, out["Combined Ratio"].between(1.00, 1.15, inclusive="both")],
        ["Good", "OK"],
        default="Poor",
    )
    out["Performance Band"] = np.select(
        [out["Performance"] >= 1.0, out["Performance"].between(0.90, 1.0, inclusive="left")],
        ["Good", "OK"],
        default="Poor",
    )

    band_score = {"Good": 1.0, "OK": 0.0, "Poor": -1.0}
    out["Actual Health Score"] = (
        out["ROE Band"].map(band_score).fillna(0)
        + out["CR Band"].map(band_score).fillna(0)
        + out["Performance Band"].map(band_score).fillna(0)
    ) / 3.0

    target_map = {
        "Strongest Momentum": 1.0,
        "Moderate Momentum": 0.6,
        "Minimal Growth": 0.2,
        "LTV Constrained": -0.4,
        "Closure Constrained": -0.4,
        "Inactive/Low Spend": -0.8,
    }
    out["Strategy Target Score"] = out["Strategy Bucket"].map(target_map).fillna(0.0)
    out["Conflict Delta"] = (out["Strategy Target Score"] - out["Actual Health Score"]).abs()

    out["Conflict Level"] = np.select(
        [out["Conflict Delta"] <= 0.35, out["Conflict Delta"] <= 0.90],
        ["Full Match", "Small Conflict"],
        default="High Conflict",
    )
    out["Conflict Arrow"] = out["Conflict Level"].map(
        {"Full Match": "⬆", "Small Conflict": "⬅", "High Conflict": "⬇"}
    )
    out["Performance Tone"] = np.select(
        [out["Actual Health Score"] > 0.2, out["Actual Health Score"] < -0.2],
        ["Good", "Poor"],
        default="OK",
    )
    out["Conflict Flag"] = np.where(out["Conflict Level"] == "Full Match", "Aligned", "Conflict")
    perf_groups = out.apply(
        lambda r: classify_perf_group(
            r.get("ROE", np.nan),
            r.get("Combined Ratio", np.nan),
            r.get("Performance", np.nan),
            r.get("Binds", np.nan),
            settings.min_binds_perf_sig,
        ),
        axis=1,
    )
    out["ROE Performance Group"] = perf_groups.map(lambda x: x[0])
    out["Performance Stat Sig"] = perf_groups.map(lambda x: x[1])
    return out


@st.cache_data(show_spinner=False, hash_funcs={Settings: lambda s: tuple(vars(s).items())})
def prepare_state_seg(state_seg_df: pd.DataFrame, state_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    seg = normalize_columns(state_seg_df).copy()
    st = normalize_columns(state_df).copy()

    for col in ["ROE", "Combined Ratio", "Performance", "CPB", "Target CPB", "Clicks", "Binds", "Avg. MRLTV", "Quotes to Binds"]:
        if col in seg.columns:
            seg[col] = to_numeric(seg[col])
        if col in st.columns:
            st[col] = to_numeric(st[col])

    seg["Clicks to Binds"] = np.where(seg["Clicks"] > 0, seg["Binds"] / seg["Clicks"], np.nan)

    st_small = st[["State", "Performance", "ROE", "Combined Ratio", "CPB", "Avg. MRLTV"]].rename(
        columns={
            "Performance": "State Performance",
            "ROE": "State ROE",
            "Combined Ratio": "State Combined Ratio",
            "CPB": "State CPB",
            "Avg. MRLTV": "State Avg. MRLTV",
        }
    )

    out = seg.merge(st_small, on="State", how="left")
    out["Perf Delta vs State"] = out["Performance"] - out["State Performance"]
    out["ROE Delta vs State"] = out["ROE"] - out["State ROE"]
    out["CR Delta vs State"] = out["Combined Ratio"] - out["State Combined Ratio"]
    seg_perf_groups = out.apply(
        lambda r: classify_perf_group(
            r.get("ROE", np.nan),
            r.get("Combined Ratio", np.nan),
            r.get("Performance", np.nan),
            r.get("Binds", np.nan),
            settings.min_binds_perf_sig,
        ),
        axis=1,
    )
    out["ROE Performance Group"] = seg_perf_groups.map(lambda x: x[0])
    out["Performance Stat Sig"] = seg_perf_groups.map(lambda x: x[1])
    return out


@st.cache_data(show_spinner=False)
def prepare_channel_state(channel_state_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(channel_state_df).copy()
    for col in [
        "Bids", "Avg. CPC", "Avg. Bid", "Impressions", "SOV", "Clicks", "Bids to Clicks",
        "ROE", "Combined Ratio", "Performance", "CPB", "Target CPB", "Quote Start Rate",
        "Clicks to Quotes", "Avg. MRLTV", "Binds", "Total Click Cost"
    ]:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    df["Segment"] = df["Channel Groups"].map(extract_segment)
    # Exclude channel groups that do not map to a known insurance segment.
    df = df[df["Segment"] != "UNKNOWN"].copy()
    return df


@st.cache_data(show_spinner=False, hash_funcs={Settings: lambda s: tuple(vars(s).items())})
def prepare_price_exploration(price_df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(price_df).copy()
    for col in ["Price Adjustment Percent", "Bids", "Avg. CPC", "Avg. Bid", "Impressions", "SOV", "Clicks"]:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    df["Win Rate"] = np.where(df["Bids"] > 0, df["Clicks"] / df["Bids"], np.nan)
    df["Test Point Stat Sig"] = (
        (df["Bids"].fillna(0) >= settings.min_bids_price_sig)
        & (df["Clicks"].fillna(0) >= settings.min_clicks_price_sig)
    )

    key_cols = ["Channel Groups"]
    if "State" in df.columns:
        key_cols = ["State", "Channel Groups"]

    base = df[df["Price Adjustment Percent"] == 0][
        key_cols + ["Clicks", "Avg. CPC", "Win Rate", "SOV", "Test Point Stat Sig"]
    ].rename(
        columns={
            "Clicks": "Baseline Clicks",
            "Avg. CPC": "Baseline CPC",
            "Win Rate": "Baseline Win Rate",
            "SOV": "Baseline SOV",
            "Test Point Stat Sig": "Baseline Stat Sig",
        }
    )

    out = df.merge(base, on=key_cols, how="left")
    out["Stat Sig Price Point"] = out["Test Point Stat Sig"].fillna(False) & out["Baseline Stat Sig"].fillna(False)
    out["Clicks Lift %"] = np.where(
        (out["Baseline Clicks"] > 0) & out["Stat Sig Price Point"],
        out["Clicks"] / out["Baseline Clicks"] - 1,
        np.nan,
    )
    out["CPC Lift %"] = np.where(
        (out["Baseline CPC"] > 0) & out["Stat Sig Price Point"],
        out["Avg. CPC"] / out["Baseline CPC"] - 1,
        np.nan,
    )
    out["Win Rate Lift %"] = np.where(
        (out["Baseline Win Rate"] > 0) & out["Stat Sig Price Point"],
        out["Win Rate"] / out["Baseline Win Rate"] - 1,
        np.nan,
    )

    out["Growth Opportunity Score"] = (
        0.70 * out["Win Rate Lift %"].fillna(0)
        + 0.30 * (1 - out["Baseline SOV"].fillna(0.5))
        - effective_cpc_penalty(settings) * np.maximum(out["CPC Lift %"].fillna(0), 0)
    )

    feasible = out[
        (out["Stat Sig Price Point"] == True)
        & (out["CPC Lift %"].fillna(0) <= effective_cpc_cap_pct(settings) / 100.0)
        & (out["Price Adjustment Percent"].fillna(0) >= 0)
        & (out["Win Rate Lift %"].fillna(0) >= 0)
    ].copy()
    # Choose test point by win-rate upside first, then growth score.
    best = feasible.sort_values(
        ["Win Rate Lift %", "Growth Opportunity Score"],
        ascending=False,
    ).groupby("Channel Groups", as_index=False).first()
    # Ensure every channel group has a default baseline candidate.
    baseline = out[out["Price Adjustment Percent"] == 0].copy()
    if not baseline.empty:
        missing = baseline[~baseline["Channel Groups"].isin(best["Channel Groups"])]
        if not missing.empty:
            best = pd.concat([best, missing], ignore_index=True)

    best["Has Sig Price Evidence"] = best["Channel Groups"].isin(
        feasible["Channel Groups"].dropna().unique().tolist()
    )
    best["Price Adjustment Percent"] = np.where(best["Has Sig Price Evidence"], best["Price Adjustment Percent"], 0.0)
    best["Growth Opportunity Score"] = np.where(best["Has Sig Price Evidence"], best["Growth Opportunity Score"], 0.0)
    best["Clicks Lift %"] = np.where(best["Has Sig Price Evidence"], best["Clicks Lift %"], 0.0)
    best["CPC Lift %"] = np.where(best["Has Sig Price Evidence"], best["CPC Lift %"], 0.0)
    best["Win Rate Lift %"] = np.where(best["Has Sig Price Evidence"], best["Win Rate Lift %"], 0.0)

    keep_cols = [
        "Channel Groups", "Price Adjustment Percent", "Growth Opportunity Score",
        "Clicks Lift %", "CPC Lift %", "Win Rate Lift %", "Has Sig Price Evidence"
    ]
    return out, best[keep_cols]


@st.cache_data(show_spinner=False, hash_funcs={Settings: lambda s: tuple(vars(s).items())})
def build_model_tables(
    state_df: pd.DataFrame,
    state_seg_df: pd.DataFrame,
    channel_state_df: pd.DataFrame,
    best_adj_df: pd.DataFrame,
    price_eval_df: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    st_ref = state_df[[
        "State", "Strategy Bucket", "Strategy Scale", "Profitability Score", "Performance", "ROE", "Combined Ratio", "Avg. MRLTV"
    ]].rename(
        columns={
            "Performance": "State Performance",
            "ROE": "State ROE",
            "Combined Ratio": "State Combined Ratio",
            "Avg. MRLTV": "State Avg. MRLTV",
        }
    )

    seg_ref = state_seg_df[[
        "State", "Segment", "Performance", "ROE", "Combined Ratio", "Avg. MRLTV", "Clicks", "Binds", "Clicks to Binds"
    ]].rename(
        columns={
            "Performance": "Seg Performance",
            "ROE": "Seg ROE",
            "Combined Ratio": "Seg Combined Ratio",
            "Avg. MRLTV": "Seg Avg. MRLTV",
            "Clicks": "Seg Clicks",
            "Binds": "Seg Binds",
            "Clicks to Binds": "Seg Clicks to Binds",
        }
    )

    rec = channel_state_df.merge(st_ref, on="State", how="left").merge(seg_ref, on=["State", "Segment"], how="left")
    rec = rec.merge(best_adj_df, on="Channel Groups", how="left")
    rec = _assign_best_price_points(rec, price_eval_df, settings)
    rec["Has Sig Price Evidence"] = rec["Has Sig Price Evidence"].fillna(False)

    rec["Use Seg Perf"] = rec["Bids"] >= settings.min_bids_channel_state
    rec["Perf Proxy"] = np.where(rec["Use Seg Perf"], rec["Seg Performance"], rec["State Performance"])
    rec["ROE Proxy"] = np.where(rec["Use Seg Perf"], rec["Seg ROE"], rec["State ROE"])
    rec["CR Proxy"] = np.where(rec["Use Seg Perf"], rec["Seg Combined Ratio"], rec["State Combined Ratio"])
    rec["MRLTV Proxy"] = np.where(rec["Use Seg Perf"], rec["Seg Avg. MRLTV"], rec["State Avg. MRLTV"])

    median_ltv = rec["MRLTV Proxy"].median(skipna=True)
    rec["Performance Score"] = (
        0.35 * rec["Perf Proxy"].fillna(0)
        + 0.30 * rec["ROE Proxy"].fillna(-0.2)
        + 0.20 * (1 - rec["CR Proxy"].fillna(1.15))
        + 0.15 * ((rec["MRLTV Proxy"].fillna(median_ltv) / median_ltv) - 1)
    )

    rec["Suggested Price Adjustment %"] = rec["Price Adjustment Percent"].fillna(0)
    rec["Growth Score"] = np.where(rec["Has Sig Price Evidence"], rec["Growth Opportunity Score"].fillna(0), 0.0)
    rec["Intent Score Raw"] = (
        0.60 * rec["Quote Start Rate"].fillna(0)
        + 0.40 * rec["Clicks to Quotes"].fillna(0)
    )
    rec["Intent Stat Sig"] = (
        (rec["Clicks"].fillna(0) >= settings.min_clicks_intent_sig)
        & rec["Quote Start Rate"].notna()
        & rec["Clicks to Quotes"].notna()
    )
    rec["Intent Score"] = np.where(rec["Intent Stat Sig"], rec["Intent Score Raw"], 0.0)

    # Binds-growth oriented score: emphasize growth + intent + strategy, while still using profitability.
    rec["Composite Score"] = (
        settings.growth_weight * rec["Growth Score"]
        + settings.profit_weight * rec["Performance Score"]
        + 0.25 * rec["Intent Score"].fillna(0)
        + 0.20 * rec["Strategy Scale"].fillna(0.5)
    )

    rec.loc[rec["Composite Score"] >= settings.aggressive_cutoff, "Suggested Price Adjustment %"] = np.maximum(
        rec["Suggested Price Adjustment %"], 20
    )
    rec.loc[
        (rec["Composite Score"] >= settings.controlled_cutoff)
        & (rec["Composite Score"] < settings.aggressive_cutoff),
        "Suggested Price Adjustment %",
    ] = np.maximum(np.minimum(rec["Suggested Price Adjustment %"], 30), 15)
    rec.loc[
        (rec["Composite Score"] >= settings.maintain_cutoff)
        & (rec["Composite Score"] < settings.controlled_cutoff),
        "Suggested Price Adjustment %",
    ] = 10
    rec.loc[rec["Composite Score"] < settings.maintain_cutoff, "Suggested Price Adjustment %"] = 5

    rec["Strategy Max Adj %"] = rec["Strategy Bucket"].apply(lambda x: strategy_max_adjustment(x, settings))
    rec["Suggested Price Adjustment %"] = np.minimum(rec["Suggested Price Adjustment %"], rec["Strategy Max Adj %"])

    rec.loc[rec["Intent Score"] < settings.min_intent_for_scale, "Suggested Price Adjustment %"] = np.minimum(
        rec["Suggested Price Adjustment %"], 10
    )
    # Pull back only when both unit economics are materially weak.
    hard_pullback = (rec["ROE Proxy"] < settings.roe_pullback_floor) & (rec["CR Proxy"] > settings.cr_pullback_ceiling)
    rec.loc[hard_pullback, "Suggested Price Adjustment %"] = -5

    # Growth-lane boost for momentum states with high intent.
    growth_lane = (
        rec["Strategy Bucket"].isin(["Strongest Momentum", "Moderate Momentum"])
        & (rec["Intent Score"] >= 0.90)
        & (rec["Growth Score"] > 0.05)
    )
    rec.loc[growth_lane, "Suggested Price Adjustment %"] = np.maximum(rec["Suggested Price Adjustment %"], 20)
    rec["Suggested Price Adjustment %"] = np.minimum(rec["Suggested Price Adjustment %"], rec["Strategy Max Adj %"])

    rec = apply_price_effects(rec, price_eval_df, settings)
    rec["Clicks Lift %"] = rec["Clicks Lift %"].fillna(0)
    rec["Win Rate Lift %"] = rec["Win Rate Lift %"].fillna(0)
    rec["CPC Lift %"] = rec["CPC Lift %"].fillna(0)
    rec.loc[~rec["Has Sig Price Evidence"], ["Suggested Price Adjustment %", "Applied Price Adjustment %", "Clicks Lift %", "Win Rate Lift %", "CPC Lift %"]] = 0.0
    # Growth is based on win-rate uplift applied to the row's bid volume.
    rec["Lift Proxy %"] = rec["Win Rate Lift %"]
    no_growth = rec["Win Rate Lift %"] < 0
    rec.loc[no_growth, "Suggested Price Adjustment %"] = 0
    rec.loc[no_growth, "Applied Price Adjustment %"] = 0
    rec.loc[no_growth, "Clicks Lift %"] = 0
    rec.loc[no_growth, "Win Rate Lift %"] = 0
    rec.loc[no_growth, "Lift Proxy %"] = 0
    rec.loc[no_growth, "CPC Lift %"] = 0

    fallback_win_rate = pd.Series(np.where(rec["Bids"] > 0, rec["Clicks"] / rec["Bids"], 0), index=rec.index)
    current_win_rate = rec["Bids to Clicks"].combine_first(fallback_win_rate)
    rec["Test-based Additional Clicks"] = np.where(
        rec["Has Sig Price Evidence"],
        rec["Bids"].fillna(0) * current_win_rate.fillna(0) * rec["Win Rate Lift %"].clip(lower=0),
        0.0,
    )
    rec["Model-based Additional Clicks"] = 0.0
    rec["Expected Additional Clicks"] = rec["Test-based Additional Clicks"].fillna(0)
    # After merges, channel-level and seg-level bind rates may coexist; use seg rate when reliable.
    channel_bind_rate = rec.get("Clicks to Binds")
    if channel_bind_rate is None:
        channel_bind_rate = np.nan
    rec["Clicks to Binds Proxy"] = np.where(rec["Use Seg Perf"], rec["Seg Clicks to Binds"], channel_bind_rate)
    rec["Clicks to Binds Proxy"] = rec["Clicks to Binds Proxy"].fillna(rec["Seg Clicks to Binds"])
    rate = rec["Clicks to Binds Proxy"]
    rec["Expected Additional Binds"] = rec["Expected Additional Clicks"] * rate.fillna(0)
    rec["Expected Additional Cost"] = (
        (rec["Clicks"] + rec["Expected Additional Clicks"]) * rec["Avg. CPC"] * (1 + rec["CPC Lift %"])
        - rec["Clicks"] * rec["Avg. CPC"]
    )

    rec["Recommendation"] = np.select(
        [
            rec["Suggested Price Adjustment %"] >= 10,
            rec["Suggested Price Adjustment %"].between(1, 9.999999),
            rec["Suggested Price Adjustment %"] == 0,
        ],
        ["Aggressive Scale", "Controlled Scale", "Maintain"],
        default="Pull Back",
    )
    rec.loc[~rec["Has Sig Price Evidence"], "Recommendation"] = "No Sig Test - Hold"

    perf_group_row = rec.apply(
        lambda r: classify_perf_group(
            r.get("ROE Proxy", np.nan),
            r.get("CR Proxy", np.nan),
            r.get("Perf Proxy", np.nan),
            r.get("Binds", np.nan),
            settings.min_binds_perf_sig,
        ),
        axis=1,
    )
    rec["Data Performance Group"] = perf_group_row.map(lambda x: x[0])
    rec["Data Performance Sig"] = perf_group_row.map(lambda x: x[1])

    state_extra = rec.groupby("State", as_index=False).agg(
        Expected_Additional_Clicks=("Expected Additional Clicks", "sum"),
        Expected_Additional_Binds=("Expected Additional Binds", "sum"),
    )

    state_seg_extra = rec.groupby(["State", "Segment"], as_index=False).agg(
        Expected_Additional_Clicks=("Expected Additional Clicks", "sum"),
        Expected_Additional_Binds=("Expected Additional Binds", "sum"),
    )

    channel_summary = rec.groupby(["Channel Groups", "Segment"], as_index=False).agg(
        Clicks=("Clicks", "sum"),
        Suggested_Price_Adjustment_pct=("Suggested Price Adjustment %", "median"),
        Expected_Additional_Clicks=("Expected Additional Clicks", "sum"),
        Expected_Additional_Binds=("Expected Additional Binds", "sum"),
        ROE=("ROE Proxy", "mean"),
        Combined_Ratio=("CR Proxy", "mean"),
        Avg_MRLTV=("MRLTV Proxy", "mean"),
        States=("State", lambda x: ", ".join(sorted(set(x)))),
        Strategy_Buckets=("Strategy Bucket", lambda x: ", ".join(sorted(set(x.dropna())))),
    )

    return rec, state_extra, state_seg_extra, channel_summary


def styled_table(df: pd.DataFrame, perf_cols: list[str], strategy_cols: list[str]):
    styler = df.style
    if perf_cols:
        styler = styler.set_properties(subset=[c for c in perf_cols if c in df.columns], **{"background-color": "#FFF3CD"})
    if strategy_cols:
        styler = styler.set_properties(subset=[c for c in strategy_cols if c in df.columns], **{"background-color": "#D9EDF7"})
    return styler


def _fmt_pct(x):
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def _fmt_cur(x):
    return "n/a" if pd.isna(x) else f"${x:,.0f}"


def _safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0)
    m = v.notna() & w.notna() & (w > 0)
    if not m.any():
        return float(v.mean(skipna=True)) if v.notna().any() else np.nan
    return float(np.average(v[m], weights=w[m]))


def simulate_mode_by_strategy(rec_df: pd.DataFrame, price_eval_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    mode_map = {
        "Strongest Momentum": "Max Growth",
        "Moderate Momentum": "Growth Leaning",
        "Minimal Growth": "Balanced",
        "LTV Constrained": "Cost Leaning",
        "Closure Constrained": "Cost Leaning",
        "Inactive/Low Spend": "Optimize Cost",
    }
    base = rec_df.copy()
    base["Strategy Mode"] = base["Strategy Bucket"].map(mode_map).fillna("Balanced")

    out_parts: list[pd.DataFrame] = []
    for mode in OPTIMIZATION_MODES:
        part = base[base["Strategy Mode"] == mode].copy()
        if part.empty:
            continue
        mode_settings = Settings(**{**settings.__dict__, "optimization_mode": mode})
        part["Scenario Target Adj %"] = part["Applied Price Adjustment %"]
        part["Scenario Target Adj %"] = np.minimum(part["Scenario Target Adj %"], part["Strategy Max Adj %"])
        part = apply_scenario_effects(part, price_eval_df, "Scenario Target Adj %", mode_settings)
        part["Scenario Lift Proxy %"] = part["Scenario Lift Proxy %"].clip(lower=0)
        wr_fb = pd.Series(np.where(part["Bids"] > 0, part["Clicks"] / part["Bids"], 0), index=part.index)
        wr = part["Bids to Clicks"].combine_first(wr_fb)
        part["Expected Additional Clicks"] = part["Bids"].fillna(0) * wr.fillna(0) * part["Scenario Lift Proxy %"]
        part["Expected Additional Binds"] = part["Expected Additional Clicks"] * part["Clicks to Binds Proxy"].fillna(0)
        part["Current Cost Sim"] = np.where(
            part["Total Click Cost"].notna(), part["Total Click Cost"], part["Clicks"] * part["Avg. CPC"]
        )
        part["Expected Total Cost Sim"] = (
            (part["Clicks"] + part["Expected Additional Clicks"]) * part["Avg. CPC"] * (1 + part["Scenario CPC Lift %"].fillna(0))
        )
        part["Additional Budget Needed Sim"] = part["Expected Total Cost Sim"] - part["Current Cost Sim"]
        out_parts.append(part)
    if not out_parts:
        return base.assign(
            **{
                "Expected Additional Clicks": 0.0,
                "Expected Additional Binds": 0.0,
                "Current Cost Sim": 0.0,
                "Expected Total Cost Sim": 0.0,
                "Additional Budget Needed Sim": 0.0,
            }
        )
    return pd.concat(out_parts, ignore_index=True)


def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_cols = {
        "ROE", "Combined Ratio", "Performance",
        "ROE Proxy", "CR Proxy", "Performance Score",
        "Clicks to Binds", "Seg Clicks to Binds", "Clicks to Binds Proxy",
        "SOV", "Bids to Clicks", "Win Rate", "CPC Lift %", "Total Cost Impact %", "Quotes to Binds", "Q2B",
        "Scenario Clicks Lift %", "Scenario Win Rate Lift %", "Scenario CPC Lift %", "Scenario Lift Proxy %",
        "Expected Performance", "Actual Performance (CPB)", "Performance Delta",
        "Intent Sig Coverage", "Price Sig Coverage",
    }
    currency_cols = {
        "Avg. MRLTV", "State Avg. MRLTV", "Seg Avg. MRLTV", "MRLTV Proxy", "Avg_LTV", "Avg_MRLTV",
        "CPB", "State CPB", "Target CPB", "Avg. CPC", "Avg. Bid", "Baseline CPC", "Expected Additional Cost",
        "Total Cost", "Expected Total Cost", "Additional Budget Required", "Additional Budget Needed",
        "Current Cost", "Actual CPB", "Expected CPB", "Target CPB (avg)",
    }

    for c in out.columns:
        if c in pct_cols and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt_pct)
        elif c in currency_cols and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt_cur)
        elif c in {"Suggested Price Adjustment %", "Applied Price Adjustment %", "Suggested_Price_Adjustment_pct", "Recommended Bid Adjustment", "Scenario Bid Adjustment %"} and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.0f}%")
    return out


def render_formatted_table(df: pd.DataFrame, use_container_width: bool = True):
    out = df.copy()
    column_config = {}

    ratio_pct_cols = {
        "ROE", "Combined Ratio", "Performance", "ROE Proxy", "CR Proxy", "Performance Score",
        "Clicks to Binds", "Seg Clicks to Binds", "Clicks to Binds Proxy", "SOV", "Bids to Clicks",
        "Win Rate", "CPC Lift %", "Total Cost Impact %", "Quotes to Binds", "Q2B",
        "Scenario Clicks Lift %", "Scenario Win Rate Lift %", "Scenario CPC Lift %", "Scenario Lift Proxy %",
        "Expected Performance", "Actual Performance (CPB)", "Performance Delta",
        "Intent Sig Coverage", "Price Sig Coverage",
    }
    point_pct_cols = {
        "Suggested Price Adjustment %", "Applied Price Adjustment %", "Suggested_Price_Adjustment_pct",
        "Recommended Bid Adjustment", "Scenario Bid Adjustment %",
    }
    currency_cols = {
        "Avg. MRLTV", "State Avg. MRLTV", "Seg Avg. MRLTV", "MRLTV Proxy", "Avg_LTV", "Avg_MRLTV",
        "CPB", "State CPB", "Target CPB", "Avg. CPC", "Avg. Bid", "Baseline CPC", "Expected Additional Cost",
        "Total Cost", "Expected Total Cost", "Additional Budget Required", "Additional Budget Needed",
        "Current Cost", "Actual CPB", "Expected CPB", "Target CPB (avg)", "CPC Impact Cost",
    }
    count_cols = {
        "Bids", "Clicks", "Binds", "Current Binds", "Expected Binds",
        "Expected Additional Clicks", "Expected Additional Binds",
        "Expected_Additional_Clicks", "Expected_Additional_Binds",
        "Additional Clicks", "Additional Binds",
        "Additional_Clicks", "Additional_Binds",
        "Rows",
    }
    precise_currency_cols = {"Avg. Bid", "Avg. CPC"}
    precise_pct_cols = {"Win Rate", "Q2B"}
    one_decimal_pct_cols = {"Clicks to Binds", "Seg Clicks to Binds", "Clicks to Binds Proxy"}

    for c in out.columns:
        if c in ratio_pct_cols and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c] * 100.0
            if c in one_decimal_pct_cols:
                column_config[c] = st.column_config.NumberColumn(c, format="%.1f%%")
            elif c in precise_pct_cols:
                column_config[c] = st.column_config.NumberColumn(c, format="%.2f%%")
            else:
                column_config[c] = st.column_config.NumberColumn(c, format="%.0f%%")
        elif c in point_pct_cols and pd.api.types.is_numeric_dtype(out[c]):
            column_config[c] = st.column_config.NumberColumn(c, format="%+.0f%%")
        elif c in currency_cols and pd.api.types.is_numeric_dtype(out[c]):
            if c in precise_currency_cols:
                column_config[c] = st.column_config.NumberColumn(c, format="dollar", step=0.01)
            else:
                column_config[c] = st.column_config.NumberColumn(c, format="dollar", step=1.0)
        elif c in count_cols and pd.api.types.is_numeric_dtype(out[c]):
            column_config[c] = st.column_config.NumberColumn(c, format="localized", step=1.0)
        elif pd.api.types.is_numeric_dtype(out[c]):
            # Default for any numeric column not explicitly configured.
            column_config[c] = st.column_config.NumberColumn(c, format="localized", step=1.0)

    try:
        st.dataframe(out, use_container_width=use_container_width, column_config=column_config)
    except Exception as exc:
        err = str(exc)
        # Guardrail for older/cached Streamlit builds that reject some printf tokens.
        if "Failed to format the number" in err or "unexpected placeholder" in err:
            safe_config = {}
            for k, cfg in column_config.items():
                if isinstance(cfg, st.column_config.NumberColumn):
                    fmt = getattr(cfg, "format", None)
                    if isinstance(fmt, str):
                        fmt = fmt.replace(",", "")
                        if fmt == "":
                            fmt = "%.0f"
                    else:
                        fmt = "%.0f"
                    safe_config[k] = st.column_config.NumberColumn(k, format=fmt)
                else:
                    safe_config[k] = cfg
            st.dataframe(out, use_container_width=use_container_width, column_config=safe_config)
        else:
            raise


def apply_scenario_effects(df: pd.DataFrame, price_eval_df: pd.DataFrame, adjustment_col: str, settings: Settings) -> pd.DataFrame:
    state_dict, effect_dict = _build_effect_dicts(price_eval_df, settings)

    def lookup(row: pd.Series) -> pd.Series:
        g = state_dict.get((str(row.get("State", "")), str(row["Channel Groups"])))
        if g is None or g.empty:
            g = effect_dict.get(str(row["Channel Groups"]))
        if g is None or g.empty:
            return pd.Series([0.0, 0.0, 0.0, 0.0])
        target = row[adjustment_col]
        up = g[g["Price Adjustment Percent"] >= target]
        near = up.sort_values("Price Adjustment Percent").iloc[0] if not up.empty else g.iloc[-1]
        return pd.Series(
            [near["Price Adjustment Percent"], near["Clicks Lift %"], near["Win Rate Lift %"], near["CPC Lift %"]]
        )

    out = df.copy()
    mapped = out.apply(lookup, axis=1)
    mapped.columns = [
        "Scenario Bid Adjustment %",
        "Scenario Clicks Lift %",
        "Scenario Win Rate Lift %",
        "Scenario CPC Lift %",
    ]
    out[[
        "Scenario Bid Adjustment %",
        "Scenario Clicks Lift %",
        "Scenario Win Rate Lift %",
        "Scenario CPC Lift %",
    ]] = mapped
    out["Scenario Lift Proxy %"] = out["Scenario Win Rate Lift %"]
    return out


def _recompute_row_metrics(row: pd.Series) -> pd.Series:
    r = row.copy()
    wr = r.get("Win Rate Lift %", np.nan)
    cpc = r.get("CPC Lift %", np.nan)
    bids = float(r.get("Bids", 0) or 0)
    clicks = float(r.get("Clicks", 0) or 0)
    avg_cpc = float(r.get("Avg. CPC", 0) or 0)
    wr_base = r.get("Bids to Clicks", np.nan)
    if pd.isna(wr_base):
        wr_base = (clicks / bids) if bids > 0 else 0.0
    wr = 0.0 if pd.isna(wr) else float(wr)
    cpc = 0.0 if pd.isna(cpc) else float(cpc)
    add_clicks = bids * float(wr_base) * max(wr, 0.0)
    c2b = r.get("Clicks to Binds Proxy", np.nan)
    if pd.isna(c2b):
        c2b = r.get("Clicks to Binds", np.nan)
    if pd.isna(c2b):
        c2b = 0.0
    add_binds = add_clicks * float(c2b)
    current_cost = r.get("Total Click Cost", np.nan)
    if pd.isna(current_cost):
        current_cost = clicks * avg_cpc
    expected_total_cost = (clicks + add_clicks) * avg_cpc * (1 + cpc)
    r["Expected Additional Clicks"] = add_clicks
    r["Expected Additional Binds"] = add_binds
    r["Expected Additional Cost"] = expected_total_cost - float(current_cost)
    return r


def stat_sig_level(bids: float, clicks: float, settings: Settings) -> tuple[str, str, float]:
    b_denom = max(float(settings.min_bids_price_sig), 1.0)
    c_denom = max(float(settings.min_clicks_price_sig), 1.0)
    score = min((float(bids) if pd.notna(bids) else 0.0) / b_denom, (float(clicks) if pd.notna(clicks) else 0.0) / c_denom)
    if score >= 2.5:
        return ("Strong", "🟢", score)
    if score >= 1.5:
        return ("Medium", "🟡", score)
    return ("Weak", "🔴", score)


@st.cache_data(show_spinner=False, hash_funcs={Settings: lambda s: tuple(vars(s).items())})
def build_adjustment_candidates(price_eval_df: pd.DataFrame, state: str, channel_group: str, settings: Settings) -> tuple[pd.DataFrame, str]:
    p = price_eval_df.copy()
    p = p[p["Channel Groups"] == channel_group].copy()
    if p.empty:
        return pd.DataFrame(), "None"
    p = p[p["Price Adjustment Percent"].fillna(0) >= 0].copy()
    p = p[p["Stat Sig Price Point"] == True].copy()
    p = p[p["CPC Lift %"].fillna(0) <= effective_cpc_cap_pct(settings) / 100.0].copy()
    if p.empty:
        return pd.DataFrame(), "None"

    src = "Channel"
    if "State" in p.columns:
        ps = p[p["State"] == state].copy()
        if not ps.empty:
            p = ps
            src = "State+Channel"
        else:
            p = (
                p.groupby(["Price Adjustment Percent"], as_index=False)
                .agg(
                    Bids=("Bids", "sum"),
                    Clicks=("Clicks", "sum"),
                    **{"Win Rate Lift %": ("Win Rate Lift %", "mean")},
                    **{"CPC Lift %": ("CPC Lift %", "mean")},
                )
                .sort_values("Price Adjustment Percent")
            )
            src = "Channel Fallback"

    if "Bids" not in p.columns:
        p["Bids"] = np.nan
    if "Clicks" not in p.columns:
        p["Clicks"] = np.nan

    sig_vals = p.apply(lambda r: stat_sig_level(r.get("Bids", 0), r.get("Clicks", 0), settings), axis=1)
    p["Sig Level"] = sig_vals.map(lambda x: x[0])
    p["Sig Icon"] = sig_vals.map(lambda x: x[1])
    p["Sig Score"] = sig_vals.map(lambda x: x[2])
    p = p[p["Sig Level"] != "Weak"].copy()
    p = p.sort_values("Price Adjustment Percent")
    return p, src


@st.cache_data(show_spinner=False, hash_funcs={Settings: lambda s: tuple(vars(s).items())})
def build_popup_card_options(
    rec_df: pd.DataFrame,
    price_eval_df: pd.DataFrame,
    state: str,
    channel_group: str,
    settings: Settings,
) -> tuple[pd.DataFrame, str]:
    candidates, source_used = build_adjustment_candidates(price_eval_df, state, channel_group, settings)
    rsel = rec_df[(rec_df["State"] == state) & (rec_df["Channel Groups"] == channel_group)].copy()
    if candidates.empty or rsel.empty:
        return pd.DataFrame(), source_used

    wr_fb = pd.Series(np.where(rsel["Bids"] > 0, rsel["Clicks"] / rsel["Bids"], 0), index=rsel.index)
    base_wr = rsel["Bids to Clicks"].combine_first(wr_fb).fillna(0)
    c2b = rsel["Clicks to Binds Proxy"].fillna(rsel.get("Clicks to Binds", 0)).fillna(0)
    cur_cost = float(np.nansum(np.where(rsel["Total Click Cost"].notna(), rsel["Total Click Cost"], rsel["Clicks"] * rsel["Avg. CPC"])))
    cur_binds = float(rsel["Binds"].fillna(0).sum())
    cur_cpb = (cur_cost / cur_binds) if cur_binds > 0 else np.nan

    out_rows = []
    for _, cr in candidates.sort_values("Price Adjustment Percent").iterrows():
        wr_l = float(cr.get("Win Rate Lift %", 0) or 0)
        cpc_l = float(cr.get("CPC Lift %", 0) or 0)
        add_clicks = float((rsel["Bids"].fillna(0) * base_wr * max(wr_l, 0)).sum())
        add_binds = float((rsel["Bids"].fillna(0) * base_wr * max(wr_l, 0) * c2b).sum())
        exp_cost = float(((rsel["Clicks"] + (rsel["Bids"] * base_wr * max(wr_l, 0))) * rsel["Avg. CPC"] * (1 + cpc_l)).sum())
        new_cpb = (exp_cost / (cur_binds + add_binds)) if (cur_binds + add_binds) > 0 else np.nan
        cpb_impact = (new_cpb / cur_cpb - 1) if pd.notna(new_cpb) and pd.notna(cur_cpb) and cur_cpb > 0 else np.nan
        out_rows.append(
            {
                "Bid Adj %": float(cr["Price Adjustment Percent"]),
                "Sig Icon": cr.get("Sig Icon", "⚪"),
                "Sig Level": cr.get("Sig Level", "n/a"),
                "Test Bids": float(cr.get("Bids", 0) or 0),
                "Win Rate Uplift": wr_l,
                "CPC Uplift": cpc_l,
                "Additional Clicks": add_clicks,
                "Additional Binds": add_binds,
                "Expected Total Cost": exp_cost,
                "Additional Budget Needed": exp_cost - cur_cost,
                "CPB Impact": cpb_impact,
            }
        )
    return pd.DataFrame(out_rows), source_used


@st.cache_data(show_spinner=False, hash_funcs={Settings: lambda s: tuple(vars(s).items())})
def precompute_popup_options_for_state(
    rec_df: pd.DataFrame,
    price_eval_df: pd.DataFrame,
    state: str,
    settings: Settings,
) -> pd.DataFrame:
    s = rec_df[rec_df["State"] == state]
    channels = sorted(s["Channel Groups"].dropna().unique().tolist())
    rows = []
    for ch in channels:
        p, src = build_popup_card_options(rec_df, price_eval_df, state, ch, settings)
        if p.empty:
            continue
        p = p.copy()
        p["Channel Groups"] = ch
        p["Source Used"] = src
        rows.append(p)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def format_adj_option_label(adj: float, click_uplift: float, cpc_uplift: float, cpb_impact: float, sig_level: str) -> str:
    cpb_txt = "n/a" if pd.isna(cpb_impact) else f"{cpb_impact:+.1%}"
    return (
        f"{adj:+.0f}%: {click_uplift:+.1%} Clicks || {cpc_uplift:+.1%} CPC || {cpb_txt} CPB "
        f"({str(sig_level).lower()} stat-sig)"
    )


def parse_adj_from_label(label: str) -> float | None:
    if not isinstance(label, str):
        return None
    m = re.match(r"\s*([+-]?\d+(?:\.\d+)?)%\s*:", label)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def apply_user_bid_overrides(rec_df: pd.DataFrame, price_eval_df: pd.DataFrame, settings: Settings, overrides: dict) -> pd.DataFrame:
    if not overrides:
        return rec_df
    rec = rec_df.copy()
    state_dict, channel_dict = _build_effect_dicts(price_eval_df, settings)
    for i, row in rec.iterrows():
        key = f"{row.get('State','')}|{row.get('Channel Groups','')}"
        o = overrides.get(key)
        if not isinstance(o, dict) or not o.get("apply", False):
            continue
        target = float(o.get("adj", row.get("Applied Price Adjustment %", 0.0)))
        g = state_dict.get((str(row.get("State", "")), str(row.get("Channel Groups", ""))))
        if g is None or g.empty:
            g = channel_dict.get(str(row.get("Channel Groups", "")))
        if g is None or g.empty:
            continue
        gg = g.copy()
        gg["dist"] = (gg["Price Adjustment Percent"] - target).abs()
        near = gg.sort_values(["dist", "Price Adjustment Percent"], ascending=[True, True]).iloc[0]
        rec.at[i, "Suggested Price Adjustment %"] = target
        rec.at[i, "Applied Price Adjustment %"] = near.get("Price Adjustment Percent", 0.0)
        rec.at[i, "Clicks Lift %"] = near.get("Clicks Lift %", 0.0)
        rec.at[i, "Win Rate Lift %"] = near.get("Win Rate Lift %", 0.0)
        rec.at[i, "CPC Lift %"] = near.get("CPC Lift %", 0.0)
        rec.at[i, "Lift Proxy %"] = rec.at[i, "Win Rate Lift %"]
        rec.at[i, "Recommendation"] = "User Applied"
        rec.loc[i] = _recompute_row_metrics(rec.loc[i])
    return rec


def summarize_from_rec(rec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_extra = rec.groupby("State", as_index=False).agg(
        Expected_Additional_Clicks=("Expected Additional Clicks", "sum"),
        Expected_Additional_Binds=("Expected Additional Binds", "sum"),
    )
    state_seg_extra = rec.groupby(["State", "Segment"], as_index=False).agg(
        Expected_Additional_Clicks=("Expected Additional Clicks", "sum"),
        Expected_Additional_Binds=("Expected Additional Binds", "sum"),
    )
    channel_summary = rec.groupby(["Channel Groups", "Segment"], as_index=False).agg(
        Clicks=("Clicks", "sum"),
        Suggested_Price_Adjustment_pct=("Suggested Price Adjustment %", "median"),
        Expected_Additional_Clicks=("Expected Additional Clicks", "sum"),
        Expected_Additional_Binds=("Expected Additional Binds", "sum"),
        ROE=("ROE Proxy", "mean"),
        Combined_Ratio=("CR Proxy", "mean"),
        Avg_MRLTV=("MRLTV Proxy", "mean"),
        States=("State", lambda x: ", ".join(sorted(set(x)))),
        Strategy_Buckets=("Strategy Bucket", lambda x: ", ".join(sorted(set(x.dropna())))),
    )
    return state_extra, state_seg_extra, channel_summary


STRATEGY_ORDER = [
    "Strongest Momentum",
    "Moderate Momentum",
    "Minimal Growth",
    "LTV Constrained",
    "Closure Constrained",
    "Inactive/Low Spend",
]


def _tier_catalog_strategy_split() -> pd.DataFrame:
    rows = []
    n = 1
    for strategy in STRATEGY_ORDER:
        for growth in ["High", "Low"]:
            for intent in ["High", "Low"]:
                rows.append(
                    {
                        "Tier Number": n,
                        "Strategy Bucket": strategy,
                        "Growth Tier": growth,
                        "Intent Tier": intent,
                        "Tier Name": f"T{n} {strategy} | {growth} Growth | {intent} Intent",
                    }
                )
                n += 1
    return pd.DataFrame(rows)


def build_strategy_tiers(rec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = rec.copy()
    t["Add Clicks per 1K Bids"] = np.where(
        t["Bids"].fillna(0) > 0,
        1000.0 * t["Expected Additional Clicks"].fillna(0) / t["Bids"].fillna(0),
        0.0,
    )
    t["Growth Tier"] = (
        t.groupby("Strategy Bucket", group_keys=False)["Add Clicks per 1K Bids"]
        .apply(lambda s: quantile_bucket(s.fillna(0), ["Low", "High"]))
    )
    t["Intent Tier"] = quantile_bucket(t["Intent Score"].fillna(0), ["Low", "High"])

    catalog = _tier_catalog_strategy_split()
    tier_map = {
        (r["Strategy Bucket"], r["Growth Tier"], r["Intent Tier"]): int(r["Tier Number"])
        for _, r in catalog.iterrows()
    }
    t["Tier Number"] = t.apply(
        lambda r: tier_map.get((r["Strategy Bucket"], r["Growth Tier"], r["Intent Tier"]), len(catalog)),
        axis=1,
    )
    t = t.merge(catalog[["Tier Number", "Tier Name"]], on="Tier Number", how="left")

    summary = (
        t.groupby(
            ["Tier Number", "Tier Name", "Strategy Bucket", "Growth Tier", "Intent Tier"],
            as_index=False,
        )
        .agg(
            Rows=("Channel Groups", "count"),
            Bids=("Bids", "sum"),
            Clicks=("Clicks", "sum"),
            Growth_per_1K_Bids=("Add Clicks per 1K Bids", "mean"),
            States=("State", lambda x: ", ".join(sorted(set(x)))),
            Sub_Channels=("Channel Groups", lambda x: ", ".join(sorted(set(x)))),
            Segments=("Segment", lambda x: ", ".join(sorted(set(x)))),
            Additional_Clicks=("Expected Additional Clicks", "sum"),
            Additional_Binds=("Expected Additional Binds", "sum"),
            Current_Binds=("Binds", "sum"),
        )
    )

    summary = catalog.merge(
        summary,
        on=["Tier Number", "Tier Name", "Strategy Bucket", "Growth Tier", "Intent Tier"],
        how="left",
    ).sort_values("Tier Number")

    fill_text = ["States", "Sub_Channels", "Segments"]
    for c in fill_text:
        summary[c] = summary[c].fillna("n/a")
    fill_num = ["Rows", "Bids", "Clicks", "Growth_per_1K_Bids", "Additional_Clicks", "Additional_Binds", "Current_Binds"]
    for c in fill_num:
        summary[c] = summary[c].fillna(0.0)
    summary = summary[summary["Rows"] > 0].copy()

    summary = summary[
        [
            "Tier Number",
            "Tier Name",
            "Strategy Bucket",
            "Growth Tier",
            "Intent Tier",
            "Rows",
            "Bids",
            "Clicks",
            "Growth_per_1K_Bids",
            "Additional_Clicks",
            "Additional_Binds",
            "Current_Binds",
            "States",
            "Sub_Channels",
            "Segments",
        ]
    ]

    detail = t[
        [
            "Tier Number",
            "Tier Name",
            "Strategy Bucket",
            "Growth Tier",
            "Intent Tier",
            "State",
            "Channel Groups",
            "Segment",
            "Bids",
            "Clicks",
            "Add Clicks per 1K Bids",
            "Expected Additional Clicks",
            "Expected Additional Binds",
            "Suggested Price Adjustment %",
            "Applied Price Adjustment %",
            "Growth Score",
            "Intent Score",
            "Composite Score",
        ]
    ].sort_values(["Tier Number", "Expected Additional Binds"], ascending=[True, False])

    return summary, detail


PERF_GROUP_ORDER = [
    "Top Performance",
    "Strong Performance",
    "Balanced",
    "Mixed Risk",
    "Weak Performance",
    "Poor Performance",
    "Low Sig - Review",
]


def _tier_catalog_performance_split() -> pd.DataFrame:
    rows = []
    n = 1
    for perf_group in PERF_GROUP_ORDER:
        for growth in ["High", "Low"]:
            for intent in ["High", "Low"]:
                rows.append(
                    {
                        "Tier Number": n,
                        "Performance Group": perf_group,
                        "Growth Tier": growth,
                        "Intent Tier": intent,
                        "Tier Name": f"T{n} {perf_group} | {growth} Growth | {intent} Intent",
                    }
                )
                n += 1
    return pd.DataFrame(rows)


def build_performance_tiers(rec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = rec.copy()
    t["Performance Group"] = t["Data Performance Group"].fillna("Low Sig - Review")
    t["Add Clicks per 1K Bids"] = np.where(
        t["Bids"].fillna(0) > 0,
        1000.0 * t["Expected Additional Clicks"].fillna(0) / t["Bids"].fillna(0),
        0.0,
    )
    t["Growth Tier"] = (
        t.groupby("Performance Group", group_keys=False)["Add Clicks per 1K Bids"]
        .apply(lambda s: quantile_bucket(s.fillna(0), ["Low", "High"]))
    )
    t["Intent Tier"] = quantile_bucket(t["Intent Score"].fillna(0), ["Low", "High"])

    catalog = _tier_catalog_performance_split()
    tier_map = {
        (r["Performance Group"], r["Growth Tier"], r["Intent Tier"]): int(r["Tier Number"])
        for _, r in catalog.iterrows()
    }
    t["Tier Number"] = t.apply(
        lambda r: tier_map.get((r["Performance Group"], r["Growth Tier"], r["Intent Tier"]), len(catalog)),
        axis=1,
    )
    t = t.merge(catalog[["Tier Number", "Tier Name"]], on="Tier Number", how="left")

    summary = (
        t.groupby(
            ["Tier Number", "Tier Name", "Performance Group", "Growth Tier", "Intent Tier"],
            as_index=False,
        )
        .agg(
            Rows=("Channel Groups", "count"),
            Bids=("Bids", "sum"),
            Clicks=("Clicks", "sum"),
            Growth_per_1K_Bids=("Add Clicks per 1K Bids", "mean"),
            States=("State", lambda x: ", ".join(sorted(set(x)))),
            Sub_Channels=("Channel Groups", lambda x: ", ".join(sorted(set(x)))),
            Segments=("Segment", lambda x: ", ".join(sorted(set(x)))),
            Additional_Clicks=("Expected Additional Clicks", "sum"),
            Additional_Binds=("Expected Additional Binds", "sum"),
            Current_Binds=("Binds", "sum"),
        )
    )
    summary = catalog.merge(
        summary,
        on=["Tier Number", "Tier Name", "Performance Group", "Growth Tier", "Intent Tier"],
        how="left",
    ).sort_values("Tier Number")
    for c in ["States", "Sub_Channels", "Segments"]:
        summary[c] = summary[c].fillna("n/a")
    for c in ["Rows", "Bids", "Clicks", "Growth_per_1K_Bids", "Additional_Clicks", "Additional_Binds", "Current_Binds"]:
        summary[c] = summary[c].fillna(0.0)
    summary = summary[summary["Rows"] > 0].copy()

    summary = summary[
        [
            "Tier Number",
            "Tier Name",
            "Performance Group",
            "Growth Tier",
            "Intent Tier",
            "Rows",
            "Bids",
            "Clicks",
            "Growth_per_1K_Bids",
            "Additional_Clicks",
            "Additional_Binds",
            "Current_Binds",
            "States",
            "Sub_Channels",
            "Segments",
        ]
    ]

    detail = t[
        [
            "Tier Number",
            "Tier Name",
            "Performance Group",
            "Growth Tier",
            "Intent Tier",
            "State",
            "Channel Groups",
            "Segment",
            "Bids",
            "Clicks",
            "Add Clicks per 1K Bids",
            "Expected Additional Clicks",
            "Expected Additional Binds",
            "Suggested Price Adjustment %",
            "Applied Price Adjustment %",
            "Growth Score",
            "Intent Score",
            "Composite Score",
        ]
    ].sort_values(["Tier Number", "Expected Additional Binds"], ascending=[True, False])

    return summary, detail


def main() -> None:
    with st.sidebar:
        dark_mode = st.toggle("Dark mode", value=True)
        fast_mode = st.toggle("Fast interaction mode", value=True, help="Reduces heavy chart rendering for faster clicks/saves.")
    st.markdown(DARK_CSS if dark_mode else LIGHT_CSS, unsafe_allow_html=True)

    plotly_template = "plotly_dark" if dark_mode else "plotly_white"

    st.title("Insurance Growth Navigator")
    st.markdown(
        """
        <div class="hero-card">
        <h4>🚀 Growth Meets Profitability</h4>
        <div class="tiny-note">🧭 State momentum map • 📈 Channel insights • 🎯 Bid adjustment recommendations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Data Paths")
        data_mode = st.radio(
            "Data source mode",
            options=["Repo data (GitHub)", "Upload files (Cloud)", "Local paths (Desktop)"],
            index=0,
        )

        strategy_upload = state_upload = state_seg_upload = None
        channel_group_upload = price_upload = channel_state_upload = None
        strategy_path = state_path = state_seg_path = ""
        channel_group_path = price_path = channel_state_path = ""

        if data_mode == "Repo data (GitHub)":
            st.caption("Using data files from repository `data/` folder.")
        elif data_mode == "Upload files (Cloud)":
            strategy_upload = st.file_uploader("State strategy file", type=None)
            state_upload = st.file_uploader("State data CSV", type=["csv"])
            state_seg_upload = st.file_uploader("State-segment CSV", type=["csv"])
            channel_group_upload = st.file_uploader("Channel group CSV", type=["csv"])
            price_upload = st.file_uploader("Price exploration CSV", type=["csv"])
            channel_state_upload = st.file_uploader("Channel group x state CSV", type=["csv"])
        else:
            strategy_path = st.text_input("State strategy", value=DEFAULT_PATHS["state_strategy"])
            state_path = st.text_input("State data", value=DEFAULT_PATHS["state_data"])
            state_seg_path = st.text_input("State-segment data", value=DEFAULT_PATHS["state_seg"])
            channel_group_path = st.text_input("Channel group data", value=DEFAULT_PATHS["channel_group"])
            price_path = st.text_input("Price exploration data", value=DEFAULT_PATHS["channel_price_exp"])
            channel_state_path = st.text_input("Channel group x state data", value=DEFAULT_PATHS["channel_state"])

        st.header("Model Controls")
        st.caption("Binds Growth Mode: calibrated to scale high-intent growth lanes and reduce fewer bids.")
        st.markdown("**Scoring Weights**")
        growth_weight = st.slider("Growth weight", 0.0, 1.0, 0.70, 0.05)
        profit_weight = st.slider("Profitability weight", 0.0, 1.0, 0.30, 0.05)

        st.markdown("**Guardrails**")
        max_cpc_increase_pct = st.slider("Max CPC increase %", 0, 40, 25, 1)
        min_bids_channel_state = st.slider("Min bids for reliable channel-state", 1, 20, 5, 1)
        cpc_penalty_weight = st.slider("CPC penalty", 0.0, 1.5, 0.65, 0.05)
        optimization_mode = st.select_slider(
            "Growth vs Cost Optimization",
            options=OPTIMIZATION_MODES,
            value="Balanced",
            help="Max Growth favors higher win-rate tests with wider CPC tolerance. Optimize Cost tightens CPC controls.",
        )
        min_intent_for_scale = st.slider("Min intent to allow positive scaling", 0.0, 1.0, 0.65, 0.01)
        roe_pullback_floor = st.slider("ROE severe pullback floor", -1.0, 0.5, -0.45, 0.01)
        cr_pullback_ceiling = st.slider("Combined ratio severe pullback ceiling", 0.8, 1.5, 1.35, 0.01)
        max_perf_drop = st.slider("Max performance drop vs current", 0.00, 0.60, 0.15, 0.01)
        min_new_performance = st.slider("Minimum new performance", 0.20, 1.50, 0.80, 0.01)

        st.markdown("**Stat Sig Rules**")
        min_clicks_intent_sig = st.slider("Min clicks for intent significance", 10, 300, 80, 5)
        min_bids_price_sig = st.slider("Min bids for price-test significance", 10, 500, 100, 10)
        min_clicks_price_sig = st.slider("Min clicks for price-test significance", 5, 200, 30, 5)
        min_binds_perf_sig = st.slider("Min binds for state performance significance", 5, 10, 8, 1)

        st.markdown("**Score Cutoffs**")
        aggressive_cutoff = st.slider("Aggressive cutoff", 0.3, 1.0, 0.40, 0.01)
        controlled_cutoff = st.slider("Controlled cutoff", 0.2, aggressive_cutoff, min(0.25, aggressive_cutoff), 0.01)
        maintain_cutoff = st.slider("Maintain cutoff", 0.0, controlled_cutoff, min(0.10, controlled_cutoff), 0.01)

        st.markdown("**Strategy Max Adjustment (%)**")
        max_adj_strongest = st.slider("Strongest Momentum cap", -10, 60, 45, 1)
        max_adj_moderate = st.slider("Moderate Momentum cap", -10, 50, 35, 1)
        max_adj_minimal = st.slider("Minimal Growth cap", -10, 40, 25, 1)
        max_adj_constrained = st.slider("Constrained / Inactive cap", -10, 30, 15, 1)

        settings = Settings(
            max_cpc_increase_pct=max_cpc_increase_pct,
            min_bids_channel_state=min_bids_channel_state,
            cpc_penalty_weight=cpc_penalty_weight,
            growth_weight=growth_weight,
            profit_weight=profit_weight,
            aggressive_cutoff=aggressive_cutoff,
            controlled_cutoff=controlled_cutoff,
            maintain_cutoff=maintain_cutoff,
            min_intent_for_scale=min_intent_for_scale,
            roe_pullback_floor=roe_pullback_floor,
            cr_pullback_ceiling=cr_pullback_ceiling,
            max_adj_strongest=max_adj_strongest,
            max_adj_moderate=max_adj_moderate,
            max_adj_minimal=max_adj_minimal,
            max_adj_constrained=max_adj_constrained,
            min_clicks_intent_sig=min_clicks_intent_sig,
            min_bids_price_sig=min_bids_price_sig,
            min_clicks_price_sig=min_clicks_price_sig,
            min_binds_perf_sig=min_binds_perf_sig,
            optimization_mode=optimization_mode,
            max_perf_drop=max_perf_drop,
            min_new_performance=min_new_performance,
        )
        st.caption(
            f"Effective CPC cap: {effective_cpc_cap_pct(settings):.0f}% | "
            f"Effective CPC penalty: {effective_cpc_penalty(settings):.2f}"
        )
        run = st.button("Refresh", type="primary")

    # Auto-run on first load; Refresh is optional for manual reruns.
    _ = run

    try:
        if data_mode == "Repo data (GitHub)":
            strategy_df = read_state_strategy(DEFAULT_PATHS["state_strategy"])
            state_raw = read_csv(DEFAULT_PATHS["state_data"])
            state_seg_raw = read_csv(DEFAULT_PATHS["state_seg"])
            _ = read_csv(DEFAULT_PATHS["channel_group"])
            price_raw = read_csv(DEFAULT_PATHS["channel_price_exp"])
            channel_state_raw = read_csv(DEFAULT_PATHS["channel_state"])
        elif data_mode == "Upload files (Cloud)":
            required_uploads = [
                strategy_upload, state_upload, state_seg_upload,
                channel_group_upload, price_upload, channel_state_upload,
            ]
            if any(x is None for x in required_uploads):
                st.error("Please upload all six files in the sidebar, then click Run.")
                return

            strategy_text = strategy_upload.getvalue().decode("utf-8", errors="ignore")
            strategy_df = parse_state_strategy_text(strategy_text)
            state_raw = pd.read_csv(state_upload)
            state_seg_raw = pd.read_csv(state_seg_upload)
            _ = pd.read_csv(channel_group_upload)
            price_raw = pd.read_csv(price_upload)
            channel_state_raw = pd.read_csv(channel_state_upload)
        else:
            strategy_df = read_state_strategy(strategy_path)
            state_raw = read_csv(state_path)
            state_seg_raw = read_csv(state_seg_path)
            _ = read_csv(channel_group_path)
            price_raw = read_csv(price_path)
            channel_state_raw = read_csv(channel_state_path)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    state_df = prepare_state(state_raw, strategy_df, settings)
    state_seg_df = prepare_state_seg(state_seg_raw, state_raw, settings)
    channel_state_df = prepare_channel_state(channel_state_raw)
    price_eval, best_adj = prepare_price_exploration(price_raw, settings)
    rec_df, state_extra_df, state_seg_extra_df, channel_summary_df = build_model_tables(
        state_df, state_seg_df, channel_state_df, best_adj, price_eval, settings
    )
    if "bid_overrides" not in st.session_state:
        st.session_state["bid_overrides"] = {}
    rec_df = apply_user_bid_overrides(rec_df, price_eval, settings, st.session_state["bid_overrides"])
    state_extra_df, state_seg_extra_df, channel_summary_df = summarize_from_rec(rec_df)

    tabs = st.tabs([
        "🏁 Tab 0: Executive State View",
        "🗺️ Tab 1: State Momentum Map",
        "📊 Tab 2: Channel Group Analysis",
        "🧠 Tab 3: Channel Group and States",
        "🌌 Neon Insights Cockpit",
    ])

    with tabs[0]:
        map_df0 = state_df.merge(state_extra_df, on="State", how="left")
        map_df0["Expected_Additional_Clicks"] = map_df0["Expected_Additional_Clicks"].fillna(0)
        map_df0["Expected_Additional_Binds"] = map_df0["Expected_Additional_Binds"].fillna(0)
        map_df0["Indicator"] = np.where(
            map_df0["Performance Tone"] == "Good",
            "🟢",
            np.where(map_df0["Performance Tone"] == "Poor", "🔴", "🟡"),
        )
        map_df0["Conflict Label"] = map_df0["Conflict Arrow"] + " " + map_df0["Conflict Level"]
        map_df0["ROE Display"] = map_df0["ROE"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
        map_df0["CR Display"] = map_df0["Combined Ratio"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
        map_df0["Perf Display"] = map_df0["Performance"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
        map_df0["LTV Display"] = map_df0["Avg. MRLTV"].map(lambda x: "n/a" if pd.isna(x) else f"${x:,.0f}")
        map_df0["Binds Display"] = map_df0["Binds"].map(lambda x: "n/a" if pd.isna(x) else f"{x:,.0f}")
        map_df0["Add Clicks Display"] = map_df0["Expected_Additional_Clicks"].map(lambda x: f"{x:,.0f}")
        map_df0["Add Binds Display"] = map_df0["Expected_Additional_Binds"].map(lambda x: f"{x:,.1f}")
        map_df0["Perf Group Display"] = map_df0["ROE Performance Group"].fillna("Low Sig - Review")
        map_df0["Conflict Perf Label"] = (
            map_df0["Performance Tone"].fillna("Unknown").astype(str)
            + " | "
            + map_df0["Conflict Level"].fillna("Unknown").astype(str)
        )
        map_df0.loc[~map_df0["Conflict Perf Label"].isin(CONFLICT_PERF_COLOR.keys()), "Conflict Perf Label"] = "Unknown | Unknown"

        sim_all = simulate_mode_by_strategy(rec_df, price_eval, settings)
        total_clicks = float(pd.to_numeric(state_df["Clicks"], errors="coerce").fillna(0).sum())
        total_binds = float(pd.to_numeric(state_df["Binds"], errors="coerce").fillna(0).sum())
        total_bids = float(pd.to_numeric(rec_df["Bids"], errors="coerce").fillna(0).sum())
        total_clicks_ch = float(pd.to_numeric(rec_df["Clicks"], errors="coerce").fillna(0).sum())
        avg_win_rate = (total_clicks_ch / total_bids) if total_bids > 0 else np.nan
        total_cost = float(pd.to_numeric(sim_all["Current Cost Sim"], errors="coerce").fillna(0).sum())
        q2b_num = pd.to_numeric(state_df.get("Quotes to Binds", np.nan), errors="coerce")
        avg_q2b = float(q2b_num.mean(skipna=True)) if q2b_num.notna().any() else np.nan
        cpb = (total_cost / total_binds) if total_binds > 0 else np.nan
        roe_w = _safe_weighted_mean(state_df["ROE"], state_df["Binds"])
        cr_w = _safe_weighted_mean(state_df["Combined Ratio"], state_df["Binds"])
        ltv_w = _safe_weighted_mean(state_df["Avg. MRLTV"], state_df["Binds"])
        add_clicks = float(pd.to_numeric(sim_all["Expected Additional Clicks"], errors="coerce").fillna(0).sum())
        add_binds = float(pd.to_numeric(sim_all["Expected Additional Binds"], errors="coerce").fillna(0).sum())
        add_budget = float(pd.to_numeric(sim_all["Additional Budget Needed Sim"], errors="coerce").fillna(0).sum())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Clicks", f"{total_clicks:,.0f}")
        k2.metric("Avg Win Rate", "n/a" if pd.isna(avg_win_rate) else f"{avg_win_rate:.1%}")
        k3.metric("Cost", f"${total_cost:,.0f}")
        k4.metric("Avg Q2B", "n/a" if pd.isna(avg_q2b) else f"{avg_q2b:.1%}")
        k5.metric("Binds", f"{total_binds:,.0f}")
        k6, k7, k8, k9 = st.columns(4)
        k6.metric("CPB", "n/a" if pd.isna(cpb) else f"${cpb:,.0f}")
        k7.metric("ROE", "n/a" if pd.isna(roe_w) else f"{roe_w:.1%}")
        k8.metric("Combined Ratio", "n/a" if pd.isna(cr_w) else f"{cr_w:.1%}")
        k9.metric("LTV", "n/a" if pd.isna(ltv_w) else f"${ltv_w:,.0f}")
        k10, k11, k12 = st.columns(3)
        k10.metric("Additional Clicks", f"{add_clicks:,.0f}")
        k11.metric("Additional Binds", f"{add_binds:,.1f}")
        k12.metric("Required Budget", f"${add_budget:,.0f}")

        map_mode0 = st.radio(
            "Map color mode",
            options=["Product Strategy", "Performance Group", "Conflict Highlight"],
            horizontal=True,
            key="tab0_map_color_mode",
        )
        if map_mode0 == "Product Strategy":
            map_color_col0, map_color_map0, map_title0 = "Strategy Bucket", STRATEGY_COLOR, "US Map: Product Strategy + State KPIs"
        elif map_mode0 == "Performance Group":
            map_color_col0, map_color_map0, map_title0 = "ROE Performance Group", PERFORMANCE_GROUP_COLOR, "US Map: Performance Group + State KPIs"
        else:
            map_color_col0, map_color_map0, map_title0 = "Conflict Perf Label", CONFLICT_PERF_COLOR, "US Map: Conflict + Performance Highlight"

        fig0 = px.choropleth(
            map_df0,
            locations="State",
            locationmode="USA-states",
            scope="usa",
            color=map_color_col0,
            color_discrete_map=map_color_map0,
            custom_data=[
                "Strategy Bucket", "Perf Group Display", "Indicator", "Conflict Label",
                "ROE Display", "CR Display", "Perf Display", "Binds Display",
                "LTV Display", "Add Clicks Display", "Add Binds Display",
            ],
            title=map_title0,
        )
        fig0.update_traces(
            hovertemplate=(
                "<b style='font-size:15px;'>%{location}</b><br>"
                "<span style='opacity:0.88;'>%{customdata[0]}</span><br>"
                "<span style='opacity:0.78;'>Perf Group: %{customdata[1]}</span><br>"
                "<span style='opacity:0.78;'>%{customdata[2]} %{customdata[3]}</span>"
                "<br>━━━━━━━━━━━━<br>"
                "<b>ROE</b> %{customdata[4]}  ·  <b>CR</b> %{customdata[5]}<br>"
                "<b>Perf</b> %{customdata[6]}  ·  <b>Binds</b> %{customdata[7]}<br>"
                "<b>Avg LTV</b> %{customdata[8]}<br>"
                "<br><b>Growth Upside</b><br>"
                "Additional Clicks: <b>%{customdata[9]}</b><br>"
                "Additional Binds: <b>%{customdata[10]}</b><extra></extra>"
            ),
        )
        fig0.update_layout(margin=dict(l=0, r=0, t=40, b=0), template=plotly_template)
        st.plotly_chart(fig0, use_container_width=True, key="state_map_tab0")

        st.markdown("**Product Strategy Sections**")
        for strat in [s for s in STRATEGY_COLOR.keys() if s in map_df0["Strategy Bucket"].dropna().unique().tolist()]:
            st_sec = state_df[state_df["Strategy Bucket"] == strat]
            rec_sec = sim_all[sim_all["Strategy Bucket"] == strat]
            if st_sec.empty:
                continue
            states_txt = ", ".join(sorted(st_sec["State"].dropna().unique().tolist()))
            s_clicks = float(pd.to_numeric(st_sec["Clicks"], errors="coerce").fillna(0).sum())
            s_binds = float(pd.to_numeric(st_sec["Binds"], errors="coerce").fillna(0).sum())
            s_bids = float(pd.to_numeric(rec_sec["Bids"], errors="coerce").fillna(0).sum())
            s_clicks_ch = float(pd.to_numeric(rec_sec["Clicks"], errors="coerce").fillna(0).sum())
            s_wr = (s_clicks_ch / s_bids) if s_bids > 0 else np.nan
            s_cost = float(pd.to_numeric(rec_sec["Current Cost Sim"], errors="coerce").fillna(0).sum())
            s_cpb = (s_cost / s_binds) if s_binds > 0 else np.nan
            s_roe = _safe_weighted_mean(st_sec["ROE"], st_sec["Binds"])
            s_cr = _safe_weighted_mean(st_sec["Combined Ratio"], st_sec["Binds"])
            s_ltv = _safe_weighted_mean(st_sec["Avg. MRLTV"], st_sec["Binds"])
            s_add_clicks = float(pd.to_numeric(rec_sec["Expected Additional Clicks"], errors="coerce").fillna(0).sum())
            s_add_binds = float(pd.to_numeric(rec_sec["Expected Additional Binds"], errors="coerce").fillna(0).sum())
            s_add_budget = float(pd.to_numeric(rec_sec["Additional Budget Needed Sim"], errors="coerce").fillna(0).sum())

            with st.container(border=True):
                st.markdown(f"**{strat}**")
                st.caption(f"States: {states_txt}")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Clicks", f"{s_clicks:,.0f}")
                c2.metric("Cost", f"${s_cost:,.0f}")
                c3.metric("Win Rate", "n/a" if pd.isna(s_wr) else f"{s_wr:.1%}")
                c4.metric("Binds", f"{s_binds:,.0f}")
                c5.metric("CPB", "n/a" if pd.isna(s_cpb) else f"${s_cpb:,.0f}")
                c6, c7, c8 = st.columns(3)
                c6.metric("ROE", "n/a" if pd.isna(s_roe) else f"{s_roe:.1%}")
                c7.metric("LTV", "n/a" if pd.isna(s_ltv) else f"${s_ltv:,.0f}")
                c8.metric("Combined Ratio", "n/a" if pd.isna(s_cr) else f"{s_cr:.1%}")
                c9, c10, c11 = st.columns(3)
                c9.metric("Additional Clicks", f"{s_add_clicks:,.0f}")
                c10.metric("Additional Binds", f"{s_add_binds:,.1f}")
                c11.metric("Required Budget", f"${s_add_budget:,.0f}")

                seg_tbl = rec_sec.groupby("Segment", as_index=False).agg(
                    Clicks=("Clicks", "sum"),
                    Bids=("Bids", "sum"),
                    Cost=("Current Cost Sim", "sum"),
                    Binds=("Binds", "sum"),
                    ROE=("ROE Proxy", "mean"),
                    Combined_Ratio=("CR Proxy", "mean"),
                    LTV=("MRLTV Proxy", "mean"),
                )
                seg_tbl["Win Rate"] = np.where(seg_tbl["Bids"] > 0, seg_tbl["Clicks"] / seg_tbl["Bids"], np.nan)
                seg_tbl["CPB"] = np.where(seg_tbl["Binds"] > 0, seg_tbl["Cost"] / seg_tbl["Binds"], np.nan)
                seg_tbl = seg_tbl[["Segment", "Clicks", "Cost", "Win Rate", "Binds", "CPB", "ROE", "LTV", "Combined_Ratio"]]
                seg_tbl = seg_tbl.rename(columns={"Combined_Ratio": "Combined Ratio"})
                render_formatted_table(seg_tbl, use_container_width=True)

    with tabs[1]:
        map_df = state_df.merge(state_extra_df, on="State", how="left")
        map_df["Expected_Additional_Clicks"] = map_df["Expected_Additional_Clicks"].fillna(0)
        map_df["Expected_Additional_Binds"] = map_df["Expected_Additional_Binds"].fillna(0)
        map_df["Indicator"] = np.where(
            map_df["Performance Tone"] == "Good",
            "🟢",
            np.where(map_df["Performance Tone"] == "Poor", "🔴", "🟡"),
        )
        map_df["Conflict Label"] = map_df["Conflict Arrow"] + " " + map_df["Conflict Level"]

        map_df["ROE Display"] = map_df["ROE"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
        map_df["CR Display"] = map_df["Combined Ratio"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
        map_df["Perf Display"] = map_df["Performance"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
        map_df["LTV Display"] = map_df["Avg. MRLTV"].map(lambda x: "n/a" if pd.isna(x) else f"${x:,.0f}")
        map_df["Binds Display"] = map_df["Binds"].map(lambda x: "n/a" if pd.isna(x) else f"{x:,.0f}")
        map_df["Add Clicks Display"] = map_df["Expected_Additional_Clicks"].map(lambda x: f"{x:,.0f}")
        map_df["Add Binds Display"] = map_df["Expected_Additional_Binds"].map(lambda x: f"{x:,.1f}")
        map_df["Perf Group Display"] = map_df["ROE Performance Group"].fillna("Low Sig - Review")
        map_df["Conflict Perf Label"] = (
            map_df["Performance Tone"].fillna("Unknown").astype(str)
            + " | "
            + map_df["Conflict Level"].fillna("Unknown").astype(str)
        )
        map_df.loc[~map_df["Conflict Perf Label"].isin(CONFLICT_PERF_COLOR.keys()), "Conflict Perf Label"] = "Unknown | Unknown"

        if not fast_mode:
            st.markdown("**Neon Summary Charts**")
            t1c1, t1c2, t1c3, t1c4 = st.columns(4)
            strat_mix = rec_df.groupby("Strategy Bucket", as_index=False)["Bids"].sum().sort_values("Bids", ascending=False)
            if not strat_mix.empty:
                fig_t1_strat = px.pie(
                    strat_mix,
                    names="Strategy Bucket",
                    values="Bids",
                    hole=0.65,
                    color="Strategy Bucket",
                    color_discrete_map=STRATEGY_COLOR,
                    template=plotly_template,
                    title="Strategy Mix",
                )
                fig_t1_strat.update_layout(margin=dict(l=0, r=0, t=35, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbeafe"))
                t1c1.plotly_chart(fig_t1_strat, use_container_width=True, key="tab1_strat_mix")
            seg_mix = rec_df.groupby("Segment", as_index=False)["Clicks"].sum().sort_values("Clicks", ascending=False)
            fig_t1_seg = px.bar(seg_mix, x="Segment", y="Clicks", template=plotly_template, title="Clicks by Segment")
            fig_t1_seg.update_traces(marker_color="#60a5fa")
            fig_t1_seg.update_layout(margin=dict(l=0, r=0, t=35, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbeafe"))
            t1c2.plotly_chart(fig_t1_seg, use_container_width=True, key="tab1_seg_mix")
            top_growth_states = map_df[["State", "Expected_Additional_Clicks"]].sort_values("Expected_Additional_Clicks", ascending=False).head(8)
            fig_t1_state_g = px.bar(top_growth_states, x="State", y="Expected_Additional_Clicks", template=plotly_template, title="Top Growth States")
            fig_t1_state_g.update_traces(marker_color="#34d399")
            fig_t1_state_g.update_layout(margin=dict(l=0, r=0, t=35, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbeafe"))
            t1c3.plotly_chart(fig_t1_state_g, use_container_width=True, key="tab1_state_growth")
            top_channels = rec_df.groupby("Channel Groups", as_index=False)["Expected Additional Binds"].sum().sort_values("Expected Additional Binds", ascending=False).head(8)
            fig_t1_ch = px.bar(top_channels, x="Channel Groups", y="Expected Additional Binds", template=plotly_template, title="Top Growth Channels")
            fig_t1_ch.update_traces(marker_color="#22d3ee")
            fig_t1_ch.update_layout(margin=dict(l=0, r=0, t=35, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbeafe"))
            t1c4.plotly_chart(fig_t1_ch, use_container_width=True, key="tab1_channel_growth")
        else:
            st.caption("Fast mode on: extra Tab 1 charts are hidden for speed.")

        map_mode = st.radio(
            "Map color mode",
            options=["Product Strategy", "Performance Group", "Conflict Highlight"],
            horizontal=True,
            key="tab1_map_color_mode",
        )
        if map_mode == "Product Strategy":
            map_color_col = "Strategy Bucket"
            map_color_map = STRATEGY_COLOR
            map_title = "US Map: Product Strategy + State KPIs"
        elif map_mode == "Performance Group":
            map_color_col = "ROE Performance Group"
            map_color_map = PERFORMANCE_GROUP_COLOR
            map_title = "US Map: Performance Group + State KPIs"
        else:
            map_color_col = "Conflict Perf Label"
            map_color_map = CONFLICT_PERF_COLOR
            map_title = "US Map: Conflict + Performance Highlight"

        fig = px.choropleth(
            map_df,
            locations="State",
            locationmode="USA-states",
            scope="usa",
            color=map_color_col,
            color_discrete_map=map_color_map,
            custom_data=[
                "Strategy Bucket",
                "Perf Group Display",
                "Indicator",
                "Conflict Label",
                "ROE Display",
                "CR Display",
                "Perf Display",
                "Binds Display",
                "LTV Display",
                "Add Clicks Display",
                "Add Binds Display",
            ],
            title=map_title,
        )
        fig.update_traces(
            hovertemplate=(
                "<b style='font-size:15px;'>%{location}</b><br>"
                "<span style='opacity:0.88;'>%{customdata[0]}</span><br>"
                "<span style='opacity:0.78;'>Perf Group: %{customdata[1]}</span><br>"
                "<span style='opacity:0.78;'>%{customdata[2]} %{customdata[3]}</span>"
                "<br>━━━━━━━━━━━━<br>"
                "<b>ROE</b> %{customdata[4]}  ·  <b>CR</b> %{customdata[5]}<br>"
                "<b>Perf</b> %{customdata[6]}  ·  <b>Binds</b> %{customdata[7]}<br>"
                "<b>Avg LTV</b> %{customdata[8]}<br>"
                "<br><b>Growth Upside</b><br>"
                "Additional Clicks: <b>%{customdata[9]}</b><br>"
                "Additional Binds: <b>%{customdata[10]}</b><extra></extra>"
            ),
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            legend_title_text="Strategy",
            template=plotly_template,
            clickmode="event+select",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(
                bgcolor="#0F172A" if dark_mode else "#F8FAFC",
                bordercolor="#334155" if dark_mode else "#CBD5E1",
                font=dict(color="#E2E8F0" if dark_mode else "#0F172A", size=13),
                align="left",
            ),
        )
        fig.update_geos(bgcolor="rgba(0,0,0,0)")

        if map_mode == "Conflict Highlight":
            st.markdown(
                """
                <div class="conflict-legend">
                  <span class="conflict-item"><span class="swatch" style="background:#166534;"></span>Good + Full Match</span>
                  <span class="conflict-item"><span class="swatch" style="background:#22C55E;"></span>Good + Small Conflict</span>
                  <span class="conflict-item"><span class="swatch" style="background:#86EFAC;"></span>Good + High Conflict</span>
                  <span class="conflict-item"><span class="swatch" style="background:#A16207;"></span>OK + Full Match</span>
                  <span class="conflict-item"><span class="swatch" style="background:#EAB308;"></span>OK + Small Conflict</span>
                  <span class="conflict-item"><span class="swatch" style="background:#FDE047;"></span>OK + High Conflict</span>
                  <span class="conflict-item"><span class="swatch" style="background:#991B1B;"></span>Poor + Full Match</span>
                  <span class="conflict-item"><span class="swatch" style="background:#EF4444;"></span>Poor + Small Conflict</span>
                  <span class="conflict-item"><span class="swatch" style="background:#FCA5A5;"></span>Poor + High Conflict</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="state_map",
            on_select="rerun",
            selection_mode="points",
        )

        state_list = sorted(map_df["State"].dropna().unique().tolist())
        default_state = state_list[0] if state_list else None
        if "selected_state_map" not in st.session_state:
            st.session_state["selected_state_map"] = default_state
        if "tab1_state_popup" not in st.session_state:
            st.session_state["tab1_state_popup"] = default_state

        if isinstance(event, dict):
            pts = event.get("selection", {}).get("points", [])
            if pts and isinstance(pts[0], dict):
                p0 = pts[0]
                clicked_state = p0.get("location")
                if not clicked_state:
                    idx = p0.get("point_index")
                    if idx is None:
                        idx = p0.get("pointNumber")
                    if isinstance(idx, int) and 0 <= idx < len(map_df):
                        clicked_state = map_df.iloc[idx]["State"]
                if not clicked_state and isinstance(p0.get("customdata"), (list, tuple)) and p0["customdata"]:
                    maybe = str(p0["customdata"][0]).strip().upper()
                    if maybe in state_list:
                        clicked_state = maybe
                if not clicked_state:
                    for k in ("hovertext", "text"):
                        maybe = p0.get(k)
                        if isinstance(maybe, str) and maybe.strip().upper() in state_list:
                            clicked_state = maybe.strip().upper()
                            break
                if clicked_state in state_list:
                    st.session_state["selected_state_map"] = clicked_state
                    st.session_state["tab1_state_popup"] = clicked_state

        # Keep selectbox value valid whenever the available state list changes.
        if state_list and st.session_state.get("tab1_state_popup") not in state_list:
            st.session_state["tab1_state_popup"] = state_list[0]
            st.session_state["selected_state_map"] = state_list[0]

        selected_state = st.selectbox(
            "Select state for detailed popup",
            options=state_list,
            index=state_list.index(st.session_state["tab1_state_popup"]) if state_list and st.session_state["tab1_state_popup"] in state_list else 0,
            key="tab1_state_popup",
        )
        st.session_state["selected_state_map"] = selected_state
        all_overrides = st.session_state.get("bid_overrides", {})
        active_global = sum(
            1 for v in all_overrides.values() if isinstance(v, dict) and v.get("apply", False)
        )
        active_state = sum(
            1
            for k, v in all_overrides.items()
            if isinstance(v, dict) and v.get("apply", False) and str(k).startswith(f"{selected_state}|")
        )
        st.caption(f"Manual Overrides Active: {active_state} in {selected_state} | {active_global} total")

        if selected_state:
            row = map_df[map_df["State"] == selected_state].head(1)
            if row.empty:
                st.warning("No state-level data found for the selected state.")
                return
            state_rows = rec_df[rec_df["State"] == selected_state].copy()
            state_current_clicks = float(row["Clicks"].iloc[0]) if pd.notna(row["Clicks"].iloc[0]) else 0.0
            state_add_clicks = float(state_rows["Expected Additional Clicks"].fillna(0).sum())
            state_add_clicks_pct = (state_add_clicks / state_current_clicks) if state_current_clicks > 0 else np.nan
            state_add_binds = float(state_rows["Expected Additional Binds"].fillna(0).sum())
            if "Total Click Cost" in state_rows.columns:
                state_current_budget = float(
                    state_rows["Total Click Cost"].fillna(state_rows["Clicks"] * state_rows["Avg. CPC"]).sum()
                )
            else:
                state_current_budget = float((state_rows["Clicks"] * state_rows["Avg. CPC"]).fillna(0).sum())
            state_add_budget = float(state_rows["Expected Additional Cost"].fillna(0).sum())
            state_add_budget_pct = (state_add_budget / state_current_budget) if state_current_budget > 0 else np.nan
            seg_view = state_seg_df[state_seg_df["State"] == selected_state].merge(
                state_seg_extra_df[state_seg_extra_df["State"] == selected_state],
                on=["State", "Segment"],
                how="left",
            )
            seg_costs = rec_df[rec_df["State"] == selected_state].groupby("Segment", as_index=False).agg(
                Bids=("Bids", "sum"),
                Clicks=("Clicks", "sum"),
                **{"Avg. CPC": ("Avg. CPC", "mean")},
                **{"Additional Budget Required": ("Expected Additional Cost", "sum")},
            )
            seg_costs["Win Rate"] = np.where(seg_costs["Bids"] > 0, seg_costs["Clicks"] / seg_costs["Bids"], np.nan)
            seg_costs = seg_costs.drop(columns=["Clicks"])
            seg_view = seg_view.merge(seg_costs, on="Segment", how="left")
            seg_view["Expected_Additional_Clicks"] = seg_view["Expected_Additional_Clicks"].fillna(0)
            seg_view["Expected_Additional_Binds"] = seg_view["Expected_Additional_Binds"].fillna(0)
            seg_view["Additional Budget Required"] = seg_view["Additional Budget Required"].fillna(0)
            if "Quotes to Binds" in seg_view.columns:
                seg_view["Q2B"] = seg_view["Quotes to Binds"]
            else:
                seg_view["Q2B"] = np.nan

            with st.container(border=True):
                st.subheader(f"🔎 State Deep Dive: {selected_state}  |  Strategy: {row['Strategy Bucket'].iloc[0]}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("💸 ROE", f"{row['ROE'].iloc[0]:.1%}")
                c2.metric("⚖️ Combined Ratio", f"{row['Combined Ratio'].iloc[0]:.1%}")
                c3.metric("🧷 Binds", f"{row['Binds'].iloc[0]:,.0f}")
                c4.metric("💎 Avg LTV", f"${row['Avg. MRLTV'].iloc[0]:,.0f}")
                tone = row["Performance Tone"].iloc[0]
                arrow = row["Conflict Arrow"].iloc[0]
                lvl = row["Conflict Level"].iloc[0]
                tone_color = "#22c55e" if tone == "Good" else "#ef4444" if tone == "Poor" else "#f59e0b"
                st.markdown(
                    f"<div style='padding:8px 10px;border-radius:10px;background:rgba(255,255,255,0.04);"
                    f"border:1px solid rgba(255,255,255,0.12);display:inline-block;'>"
                    f"<span style='font-weight:700;color:{tone_color};'>{arrow} {lvl}</span>"
                    f" <span style='color:#9ca3af;'>Strategy vs actual performance</span></div>",
                    unsafe_allow_html=True,
                )

                c5, c6, c7, c8, c9 = st.columns(5)
                c5.metric("🖱️ Current Clicks", f"{state_current_clicks:,.0f}")
                c6.metric(
                    "✨ Additional Clicks",
                    f"{state_add_clicks:,.0f}",
                    delta="n/a" if pd.isna(state_add_clicks_pct) else f"{state_add_clicks_pct:.1%}",
                )
                c7.metric("🎉 Additional Binds", f"{state_add_binds:,.1f}")
                c8.metric("💰 Additional Budget Needed", f"${state_add_budget:,.0f}")
                c9.metric("📊 Budget Impact", "n/a" if pd.isna(state_add_budget_pct) else f"{state_add_budget_pct:.1%}")

                st.markdown("**🧩 Per-Segment KPI + Opportunity**")
                seg_show = seg_view[[
                    "Segment", "ROE Performance Group",
                    "Bids", "Avg. CPC", "Win Rate", "Q2B", "Clicks", "Binds", "Clicks to Binds", "ROE", "Combined Ratio", "Avg. MRLTV",
                    "Expected_Additional_Clicks", "Expected_Additional_Binds", "Additional Budget Required"
                ]].sort_values("Expected_Additional_Clicks", ascending=False)
                render_formatted_table(seg_show, use_container_width=True)

                st.markdown("**📌 Channel Groups In This State**")
                state_channels = rec_df[rec_df["State"] == selected_state].copy()
                if state_channels.empty:
                    st.info("No channel-group rows found for this state.")
                else:
                    seg_options = sorted(state_channels["Segment"].dropna().unique().tolist())
                    selected_seg = st.multiselect(
                        "Filter channel groups by segment",
                        options=seg_options,
                        default=seg_options,
                        key="tab1_cg_segment_filter",
                    )
                    state_channels = state_channels[state_channels["Segment"].isin(selected_seg)]
                    if state_channels.empty:
                        st.info("No channel groups for selected segment filter.")
                    else:
                        # Use already-applied model outputs (including manual popup overrides)
                        # so table values are fully aligned with selected adjustments.
                        if "Total Click Cost" in state_channels.columns:
                            state_channels["Total Cost"] = state_channels["Total Click Cost"]
                        else:
                            state_channels["Total Cost"] = state_channels["Clicks"] * state_channels["Avg. CPC"]
                        state_channels["Additional Budget Needed"] = state_channels["Expected Additional Cost"].fillna(0)
                        state_channels["Expected Total Cost"] = state_channels["Total Cost"] + state_channels["Additional Budget Needed"]

                        cg_state = state_channels.groupby("Channel Groups", as_index=False).agg(
                            Bids=("Bids", "sum"),
                            Binds=("Binds", "sum"),
                            SOV=("SOV", "mean"),
                            Clicks=("Clicks", "sum"),
                            **{"Win Rate": ("Bids to Clicks", "mean")},
                            **{"Total Cost": ("Total Cost", "sum")},
                            **{"Expected Total Cost": ("Expected Total Cost", "sum")},
                            **{"Additional Budget Needed": ("Additional Budget Needed", "sum")},
                            **{"Rec. Bid Adj.": ("Applied Price Adjustment %", "median")},
                            **{"Expected Additional Clicks": ("Expected Additional Clicks", "sum")},
                            **{"Expected Additional Binds": ("Expected Additional Binds", "sum")},
                            **{"CPC Lift %": ("CPC Lift %", "mean")},
                        ).sort_values("Expected Additional Clicks", ascending=False)
                        cg_state["Total Cost Impact %"] = np.where(
                            cg_state["Total Cost"] > 0,
                            cg_state["Additional Budget Needed"] / cg_state["Total Cost"],
                            0,
                        )
                        cg_state_cols = [
                            "Channel Groups",
                            "Bids",
                            "SOV",
                            "Clicks",
                            "Binds",
                            "Rec. Bid Adj.",
                            "Win Rate",
                            "Total Cost",
                            "Expected Total Cost",
                            "Additional Budget Needed",
                            "Expected Additional Clicks",
                            "Expected Additional Binds",
                            "CPC Lift %",
                        ]
                        cg_state_cols = [c for c in cg_state_cols if c in cg_state.columns]
                        table_df = cg_state[cg_state_cols].copy()
                        table_df["Selected Price Adj."] = table_df["Rec. Bid Adj."]
                        table_df["Select"] = False
                        table_df["Apply"] = False
                        table_df["Selection Source"] = "Suggested"
                        popup_state_df = precompute_popup_options_for_state(rec_df, price_eval, selected_state, settings)
                        table_df["Adj Selection"] = ""
                        table_df["Adj Options"] = [[] for _ in range(len(table_df))]
                        table_df["Adj Options JSON"] = ["[]" for _ in range(len(table_df))]
                        for idx, rr in table_df.iterrows():
                            okey = f"{selected_state}|{rr['Channel Groups']}"
                            ov = st.session_state["bid_overrides"].get(okey, {})
                            if isinstance(ov, dict) and ov.get("apply", False):
                                table_df.at[idx, "Apply"] = True
                                table_df.at[idx, "Selected Price Adj."] = float(ov.get("adj", rr["Rec. Bid Adj."]))
                                table_df.at[idx, "Selection Source"] = "Manual"
                            ch = str(rr["Channel Groups"])
                            ch_opts = popup_state_df[popup_state_df["Channel Groups"] == ch] if not popup_state_df.empty else pd.DataFrame()
                            labels = []
                            for _, op in ch_opts.iterrows():
                                labels.append(
                                    format_adj_option_label(
                                        float(op.get("Bid Adj %", 0) or 0),
                                        float(op.get("Win Rate Uplift", 0) or 0),
                                        float(op.get("CPC Uplift", 0) or 0),
                                        op.get("CPB Impact", np.nan),
                                        str(op.get("Sig Level", "")),
                                    )
                                )
                            if not labels:
                                base_adj = float(table_df.at[idx, "Selected Price Adj."])
                                labels = [f"{base_adj:+.0f}%: n/a Clicks || n/a CPC || n/a CPB (no stat-sig)"]
                            table_df.at[idx, "Adj Options"] = labels
                            table_df.at[idx, "Adj Options JSON"] = json.dumps(labels)
                            current_adj = float(table_df.at[idx, "Selected Price Adj."])
                            selected_label = next((lb for lb in labels if parse_adj_from_label(lb) == current_adj), labels[0])
                            table_df.at[idx, "Adj Selection"] = selected_label
                        table_df = table_df[
                            [
                                "Channel Groups",
                                "Select",
                                "Bids",
                                "SOV",
                                "Clicks",
                                "Binds",
                                "Win Rate",
                                "Total Cost",
                                "Rec. Bid Adj.",
                                "Adj Selection",
                                "Selected Price Adj.",
                                "Expected Total Cost",
                                "Additional Budget Needed",
                                "Expected Additional Clicks",
                                "Expected Additional Binds",
                                "CPC Lift %",
                                "Apply",
                                "Selection Source",
                                "Adj Options",
                                "Adj Options JSON",
                            ]
                        ]
                        for c in [
                            "Bids", "SOV", "Clicks", "Binds", "Win Rate", "Total Cost",
                            "Rec. Bid Adj.", "Selected Price Adj.", "Expected Total Cost",
                            "Additional Budget Needed", "Expected Additional Clicks",
                            "Expected Additional Binds", "CPC Lift %",
                        ]:
                            if c in table_df.columns:
                                table_df[c] = pd.to_numeric(table_df[c], errors="coerce").fillna(0.0)

                        # Use dot-free internal field names for AG Grid reliability.
                        grid_df = table_df.rename(
                            columns={
                                "Rec. Bid Adj.": "Rec Bid Adj",
                                "Selected Price Adj.": "Selected Price Adj",
                            }
                        )

                        selected_groups: list[str] = []
                        edited = grid_df.copy()
                        draft_key = f"tab1_grid_draft_{selected_state}"
                        if AGGRID_AVAILABLE:
                            gb = GridOptionsBuilder.from_dataframe(edited)
                            gb.configure_default_column(resizable=True, sortable=True, filter=True)
                            gb.configure_grid_options(singleClickEdit=True, stopEditingWhenCellsLoseFocus=True)
                            gb.configure_selection("multiple", use_checkbox=True)
                            gb.configure_column("Channel Groups", editable=False, pinned="left", width=180)
                            gb.configure_column("Select", editable=True, width=64)
                            gb.configure_column("Bids", editable=False, width=86, type=["numericColumn"], valueFormatter="value == null ? '' : Math.round(value).toLocaleString()")
                            gb.configure_column("SOV", editable=False, width=76, type=["numericColumn"], valueFormatter="value == null ? '' : (value * 100).toFixed(0) + '%'")
                            gb.configure_column("Clicks", editable=False, width=84, type=["numericColumn"], valueFormatter="value == null ? '' : Math.round(value).toLocaleString()")
                            gb.configure_column("Binds", editable=False, width=84, type=["numericColumn"], valueFormatter="value == null ? '' : Number(value).toFixed(2)")
                            gb.configure_column("Win Rate", editable=False, width=94, type=["numericColumn"], valueFormatter="value == null ? '' : (value * 100).toFixed(2) + '%'")
                            gb.configure_column("Total Cost", editable=False, width=110, type=["numericColumn"], valueFormatter="value == null ? '' : '$' + Math.round(value).toLocaleString()")
                            gb.configure_column("Rec Bid Adj", headerName="Rec. Bid Adj.", editable=False, width=102, type=["numericColumn"], valueFormatter="value == null ? '' : (value>=0?'+':'') + Number(value).toFixed(0) + '%'")
                            gb.configure_column(
                                "Adj Selection",
                                headerName="Adj Selection ▼",
                                editable=True,
                                width=350,
                                valueFormatter="value ? ('▼ ' + value) : '▼ Select adjustment'",
                                cellEditor="agSelectCellEditor",
                                cellEditorParams=JsCode(
                                    """
                                    function(params) {
                                      try {
                                        const raw = (params.data && params.data['Adj Options JSON']) ? params.data['Adj Options JSON'] : '[]';
                                        const vals = JSON.parse(raw);
                                        return { values: Array.isArray(vals) ? vals : [] };
                                      } catch (e) {
                                        return { values: [] };
                                      }
                                    }
                                    """
                                ),
                            )
                            gb.configure_column("Selected Price Adj", headerName="Selected Price Adj.", editable=False, width=138, type=["numericColumn"], valueFormatter="value == null ? '' : (value>=0?'+':'') + Number(value).toFixed(0) + '%'")
                            gb.configure_column("Expected Total Cost", editable=False, width=128, type=["numericColumn"], valueFormatter="value == null ? '' : '$' + Math.round(value).toLocaleString()")
                            gb.configure_column("Additional Budget Needed", header_name="Adjusted Budget", editable=False, width=118, type=["numericColumn"], valueFormatter="value == null ? '' : '$' + Math.round(value).toLocaleString()")
                            gb.configure_column("Expected Additional Clicks", editable=False, width=126, type=["numericColumn"], valueFormatter="(value == null || isNaN(Number(value))) ? '0' : Math.round(Number(value)).toLocaleString()")
                            gb.configure_column("Expected Additional Binds", editable=False, width=124, type=["numericColumn"], valueFormatter="value == null ? '' : Number(value).toFixed(2)")
                            gb.configure_column("CPC Lift %", editable=False, width=88, type=["numericColumn"], valueFormatter="value == null ? '' : (value * 100).toFixed(0) + '%'")
                            gb.configure_column("Apply", editable=True, width=70)
                            gb.configure_column("Selection Source", editable=False, width=108)
                            gb.configure_column("Adj Options", hide=True)
                            gb.configure_column("Adj Options JSON", hide=True)
                            go = gb.build()
                            custom_css = {
                                ".ag-root-wrapper": {"background-color": "#0b1220", "border": "1px solid #1f2937", "border-radius": "10px"},
                                ".ag-header": {"background-color": "#0f172a !important", "color": "#cbd5e1"},
                                ".ag-row": {"background-color": "#0b1220", "color": "#e2e8f0"},
                                ".ag-row-hover": {"background-color": "#111827 !important"},
                                ".ag-cell": {"border-color": "#1f2937"},
                                ".ag-theme-balham-dark .ag-input-field-input": {"background-color": "#111827", "color": "#e5e7eb"},
                            }
                            grid = AgGrid(
                                edited,
                                gridOptions=go,
                                allow_unsafe_jscode=True,
                                update_mode=GridUpdateMode.VALUE_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                                fit_columns_on_grid_load=True,
                                reload_data=True,
                                height=460,
                                theme="balham-dark" if dark_mode else "balham",
                                custom_css=custom_css if dark_mode else None,
                                key=f"tab1_aggrid_{selected_state}",
                            )
                            edited = pd.DataFrame(grid["data"])
                            st.session_state[draft_key] = edited
                            selected_rows = grid.get("selected_rows", [])
                            if isinstance(selected_rows, list) and selected_rows:
                                selected_groups = [str(r.get("Channel Groups")) for r in selected_rows if r.get("Channel Groups") is not None]
                        else:
                            all_adj_options = sorted({x for xs in edited["Adj Options"].tolist() for x in (xs if isinstance(xs, list) else [])})
                            edited = st.data_editor(
                                edited,
                                use_container_width=True,
                                hide_index=True,
                                key=f"tab1_apply_editor_{selected_state}",
                                column_config={
                                    "Channel Groups": st.column_config.TextColumn("Channel Groups", disabled=True),
                                    "Select": st.column_config.CheckboxColumn("Select"),
                                    "Bids": st.column_config.NumberColumn("Bids", format="localized", disabled=True),
                                    "SOV": st.column_config.NumberColumn("SOV", format="%.0f%%", disabled=True),
                                    "Clicks": st.column_config.NumberColumn("Clicks", format="localized", disabled=True),
                                    "Binds": st.column_config.NumberColumn("Binds", format="%.2f", disabled=True),
                                    "Win Rate": st.column_config.NumberColumn("Win Rate", format="%.2f%%", disabled=True),
                                    "Total Cost": st.column_config.NumberColumn("Total Cost", format="dollar", disabled=True),
                                    "Rec Bid Adj": st.column_config.NumberColumn("Rec. Bid Adj.", format="%+.0f%%", disabled=True),
                                    "Adj Selection": st.column_config.SelectboxColumn("Adj Selection", options=all_adj_options),
                                    "Selected Price Adj": st.column_config.NumberColumn("Selected Price Adj.", format="%+.0f%%", disabled=True),
                                    "Expected Total Cost": st.column_config.NumberColumn("Expected Total Cost", format="dollar", disabled=True),
                                    "Additional Budget Needed": st.column_config.NumberColumn("Adjusted Budget", format="dollar", disabled=True),
                                    "Expected Additional Clicks": st.column_config.NumberColumn("Expected Additional Clicks", format="localized", disabled=True),
                                    "Expected Additional Binds": st.column_config.NumberColumn("Expected Additional Binds", format="%.2f", disabled=True),
                                    "CPC Lift %": st.column_config.NumberColumn("CPC Lift %", format="%.0f%%", disabled=True),
                                    "Apply": st.column_config.CheckboxColumn("Apply"),
                                    "Selection Source": st.column_config.TextColumn("Selection Source", disabled=True),
                                    "Adj Options": None,
                                    "Adj Options JSON": None,
                                },
                            )
                            st.session_state[draft_key] = edited
                            selected_rows = edited[edited["Select"] == True] if "Select" in edited.columns else pd.DataFrame()
                            if not selected_rows.empty:
                                selected_groups = selected_rows["Channel Groups"].astype(str).tolist()
                        # Stage dropdown selections in rows (apply on Save).
                        for i, rr in edited.iterrows():
                            adj = parse_adj_from_label(rr.get("Adj Selection", ""))
                            if adj is not None:
                                edited.at[i, "Selected Price Adj"] = float(adj)
                                edited.at[i, "Selection Source"] = "Manual"
                                edited.at[i, "Apply"] = True
                        st.session_state[draft_key] = edited
                        selected_rows = edited[edited["Select"] == True] if "Select" in edited.columns else pd.DataFrame()
                        if not selected_rows.empty:
                            selected_groups = selected_rows["Channel Groups"].astype(str).tolist()

                        a1, a2, a3, a4 = st.columns([1.2, 1, 1, 1])
                        bulk_adj = a1.number_input(
                            "Set selected to bid adj %",
                            min_value=-10.0,
                            max_value=60.0,
                            value=10.0,
                            step=5.0,
                            key=f"tab1_bulk_adj_{selected_state}",
                        )
                        do_apply_bulk = a2.button("Apply Selected", key=f"tab1_apply_selected_{selected_state}")
                        do_revert_bulk = a3.button("Revert Selected", key=f"tab1_revert_selected_{selected_state}")
                        do_save = a4.button("Save Edits", key=f"tab1_save_edits_{selected_state}")

                        if do_apply_bulk and selected_groups:
                            for cg in selected_groups:
                                m = edited["Channel Groups"] == cg
                                edited.loc[m, "Selected Price Adj"] = float(bulk_adj)
                                edited.loc[m, "Apply"] = True
                                edited.loc[m, "Selection Source"] = "Manual"
                            st.session_state[draft_key] = edited
                            st.rerun()
                        if do_revert_bulk and selected_groups:
                            for cg in selected_groups:
                                m = edited["Channel Groups"] == cg
                                edited.loc[m, "Apply"] = False
                                edited.loc[m, "Selection Source"] = "Suggested"
                            st.session_state[draft_key] = edited
                            st.rerun()
                        if do_save:
                            new_overrides = dict(st.session_state["bid_overrides"])
                            for _, rr in edited.iterrows():
                                okey = f"{selected_state}|{rr['Channel Groups']}"
                                adj_from_dropdown = parse_adj_from_label(rr.get("Adj Selection", ""))
                                if adj_from_dropdown is not None:
                                    new_overrides[okey] = {"apply": True, "adj": float(adj_from_dropdown)}
                                elif bool(rr.get("Apply", False)):
                                    new_overrides[okey] = {"apply": True, "adj": float(rr.get("Selected Price Adj", 0.0))}
                                else:
                                    new_overrides.pop(okey, None)
                            st.session_state["bid_overrides"] = new_overrides
                            st.session_state.pop(f"tab1_grid_draft_{selected_state}", None)
                            st.rerun()

                        st.caption("Use `Adj Selection` dropdown in the table, then click `Save Edits` to apply all changes.")

        st.markdown("**State Strategy vs Actual Indicator**")
        indicator_view = map_df[[
            "State", "Strategy Bucket", "Conflict Arrow", "Conflict Level", "Performance Tone",
            "ROE Performance Group", "Performance Stat Sig",
            "ROE", "Combined Ratio", "Performance", "Binds", "Quotes to Binds", "CPB", "Avg. MRLTV"
        ]].sort_values(["Conflict Level", "State"])
        indicator_view["Indicator"] = np.where(
            indicator_view["Performance Tone"] == "Good",
            "🟢",
            np.where(indicator_view["Performance Tone"] == "Poor", "🔴", "🟡"),
        )
        indicator_view["Match"] = indicator_view["Indicator"] + " " + indicator_view["Conflict Arrow"] + " " + indicator_view["Conflict Level"]
        indicator_view["Q2B"] = indicator_view["Quotes to Binds"]
        render_formatted_table(
            indicator_view[["State", "Strategy Bucket", "ROE Performance Group", "Performance Stat Sig", "Match", "ROE", "Combined Ratio", "Performance", "Binds", "Q2B", "CPB", "Avg. MRLTV"]],
            use_container_width=True,
        )
        st.markdown("**ROE-Based State Performance Layer**")
        state_perf_layer = (
            map_df.groupby(["ROE Performance Group", "Performance Stat Sig"], as_index=False)
            .agg(
                States=("State", lambda x: ", ".join(sorted(set(x)))),
                Rows=("State", "count"),
                Binds=("Binds", "sum"),
            )
            .sort_values(["Performance Stat Sig", "Rows"], ascending=[False, False])
        )
        render_formatted_table(state_perf_layer, use_container_width=True)

    with tabs[2]:
        st.subheader("📊 Channel Group Analysis")
        current_binds = rec_df["Binds"].fillna(0).sum()
        add_binds = rec_df["Expected Additional Binds"].fillna(0).sum()
        expected_total_binds = current_binds + add_binds
        binds_multiplier = expected_total_binds / current_binds if current_binds > 0 else np.nan
        d1, d2, d3 = st.columns(3)
        d1.metric("Current Binds", f"{current_binds:,.0f}")
        d2.metric("Expected Additional Binds", f"{add_binds:,.0f}")
        d3.metric("Projected Binds Multiplier", "n/a" if pd.isna(binds_multiplier) else f"{binds_multiplier:.2f}x")
        if not pd.isna(binds_multiplier):
            st.caption(f"Progress to 2.0x binds goal: {min(100, max(0, binds_multiplier / 2.0 * 100)):.1f}%")

        f_col1, f_col2, f_col3 = st.columns(3)
        states = sorted(rec_df["State"].dropna().unique().tolist())
        strategies = sorted(rec_df["Strategy Bucket"].dropna().unique().tolist())
        segments = sorted(rec_df["Segment"].dropna().unique().tolist())

        sel_states = f_col1.multiselect("State", options=states, default=states, key="tab2_state")
        sel_strategies = f_col2.multiselect("State Strategy", options=strategies, default=strategies, key="tab2_strategy")
        sel_segments = f_col3.multiselect("Segment", options=segments, default=segments, key="tab2_segment")

        filt = rec_df[
            rec_df["State"].isin(sel_states)
            & rec_df["Strategy Bucket"].isin(sel_strategies)
            & rec_df["Segment"].isin(sel_segments)
        ]
        intent_sig_share = filt["Intent Stat Sig"].mean() if not filt.empty and "Intent Stat Sig" in filt.columns else 0.0
        price_sig_share = filt["Has Sig Price Evidence"].mean() if not filt.empty and "Has Sig Price Evidence" in filt.columns else 0.0
        st.caption(
            f"Stat-sig coverage in filter: Intent {intent_sig_share:.0%} | Price tests {price_sig_share:.0%}"
        )
        aggr_factor = st.slider(
            "Bid aggressiveness (scenario multiplier)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="tab2_aggr_factor",
            help="Scales recommended bid adjustment before applying tested price-exploration effects.",
        )

        scen = filt.copy()
        scen["Scenario Target Adj %"] = scen["Applied Price Adjustment %"] * aggr_factor
        scen["Scenario Target Adj %"] = np.minimum(scen["Scenario Target Adj %"], scen["Strategy Max Adj %"])
        scen = apply_scenario_effects(scen, price_eval, "Scenario Target Adj %", settings)
        scen["Scenario Lift Proxy %"] = scen["Scenario Lift Proxy %"].clip(lower=0)
        scen_wr_fallback = pd.Series(
            np.where(scen["Bids"] > 0, scen["Clicks"] / scen["Bids"], 0),
            index=scen.index,
        )
        scen_wr = scen["Bids to Clicks"].combine_first(scen_wr_fallback)
        scen["Additional Clicks (scenario)"] = scen["Bids"].fillna(0) * scen_wr.fillna(0) * scen["Scenario Lift Proxy %"]
        scen["Additional Binds (scenario)"] = scen["Additional Clicks (scenario)"] * scen["Clicks to Binds Proxy"].fillna(0)
        if "Total Click Cost" in scen.columns:
            scen["Current Cost"] = scen["Total Click Cost"].fillna(scen["Clicks"] * scen["Avg. CPC"])
        else:
            scen["Current Cost"] = scen["Clicks"] * scen["Avg. CPC"]
        scen["Expected Total Cost"] = (
            (scen["Clicks"] + scen["Additional Clicks (scenario)"]) * scen["Avg. CPC"] * (1 + scen["Scenario CPC Lift %"].fillna(0))
        )
        scen["Additional Budget Needed"] = scen["Expected Total Cost"] - scen["Current Cost"]

        grp = scen.groupby("Channel Groups", as_index=False).agg(
            Bids=("Bids", "sum"),
            Clicks=("Clicks", "sum"),
            Current_Binds=("Binds", "sum"),
            Current_Cost=("Current Cost", "sum"),
            Scenario_Bid_Adjustment=("Scenario Bid Adjustment %", "median"),
            Scenario_CPC_Lift=("Scenario CPC Lift %", "mean"),
            Additional_Clicks=("Additional Clicks (scenario)", "sum"),
            Additional_Binds=("Additional Binds (scenario)", "sum"),
            Additional_Budget_Needed=("Additional Budget Needed", "sum"),
            Expected_Total_Cost=("Expected Total Cost", "sum"),
            Target_CPB=("Target CPB", "mean"),
            Intent_Sig_Coverage=("Intent Stat Sig", "mean"),
            Price_Sig_Coverage=("Has Sig Price Evidence", "mean"),
        )
        grp["Avg. CPC"] = np.where(grp["Clicks"] > 0, grp["Current_Cost"] / grp["Clicks"], np.nan)
        grp["Win Rate"] = np.where(grp["Bids"] > 0, grp["Clicks"] / grp["Bids"], np.nan)
        grp["Expected Binds"] = grp["Current_Binds"] + grp["Additional_Binds"]
        grp["Actual CPB"] = np.where(grp["Current_Binds"] > 0, grp["Current_Cost"] / grp["Current_Binds"], np.nan)
        grp["Expected CPB"] = np.where(grp["Expected Binds"] > 0, grp["Expected_Total_Cost"] / grp["Expected Binds"], np.nan)
        grp["Expected Performance"] = np.where(grp["Expected CPB"] > 0, grp["Target_CPB"] / grp["Expected CPB"], np.nan)
        grp["Actual Performance (CPB)"] = np.where(grp["Actual CPB"] > 0, grp["Target_CPB"] / grp["Actual CPB"], np.nan)
        grp["Performance Delta"] = grp["Expected Performance"] - grp["Actual Performance (CPB)"]
        grp["Total Cost Impact %"] = np.where(grp["Current_Cost"] > 0, grp["Additional_Budget_Needed"] / grp["Current_Cost"], np.nan)

        grp = grp.rename(
            columns={
                "Current_Cost": "Current Cost",
                "Scenario_Bid_Adjustment": "Scenario Bid Adjustment %",
                "Scenario_CPC_Lift": "Scenario CPC Lift %",
                "Additional_Budget_Needed": "Additional Budget Needed",
                "Expected_Total_Cost": "Expected Total Cost",
                "Target_CPB": "Target CPB (avg)",
                "Current_Binds": "Current Binds",
                "Additional_Clicks": "Additional Clicks",
                "Additional_Binds": "Additional Binds",
                "Intent_Sig_Coverage": "Intent Sig Coverage",
                "Price_Sig_Coverage": "Price Sig Coverage",
            }
        )
        show_cols = [
            "Channel Groups", "Bids", "Clicks", "Avg. CPC", "Current Cost", "Win Rate",
            "Intent Sig Coverage", "Price Sig Coverage",
            "Scenario Bid Adjustment %", "Scenario CPC Lift %",
            "Additional Clicks", "Additional Binds", "Additional Budget Needed",
            "Expected CPB", "Target CPB (avg)", "Actual CPB",
            "Expected Performance", "Actual Performance (CPB)", "Performance Delta", "Total Cost Impact %",
        ]
        show_cols = [c for c in show_cols if c in grp.columns]
        grp = grp[show_cols].sort_values("Additional Binds", ascending=False)
        render_formatted_table(grp, use_container_width=True)

    with tabs[3]:
        st.subheader("🧠 Channel Group + State Recommendations")

        c1, c2, c3, c4, c5 = st.columns(5)
        states = sorted(rec_df["State"].dropna().unique().tolist())
        strategies = sorted(rec_df["Strategy Bucket"].dropna().unique().tolist())
        segments = sorted(rec_df["Segment"].dropna().unique().tolist())
        channels = sorted(rec_df["Channel Groups"].dropna().unique().tolist())

        fs = c1.multiselect("State", options=states, default=states, key="tab3_state")
        fst = c2.multiselect("State Strategy", options=strategies, default=strategies, key="tab3_strategy")
        fseg = c3.multiselect("Segment", options=segments, default=segments, key="tab3_segment")
        fch = c4.multiselect("Channel Group", options=channels, default=channels, key="tab3_channel")
        score_min = c5.slider("Min composite score", 0.0, 1.0, 0.0, 0.05)

        out = rec_df[
            rec_df["State"].isin(fs)
            & rec_df["Strategy Bucket"].isin(fst)
            & rec_df["Segment"].isin(fseg)
            & rec_df["Channel Groups"].isin(fch)
            & (rec_df["Composite Score"] >= score_min)
        ].copy()

        show_cols = [
            "Channel Groups", "Segment", "State", "Strategy Bucket", "Bids", "Clicks", "Suggested Price Adjustment %",
            "Applied Price Adjustment %",
            "Intent Stat Sig", "Has Sig Price Evidence",
            "Expected Additional Clicks", "Expected Additional Binds", "Expected Additional Cost", "Growth Score",
            "Performance Score", "Composite Score", "ROE Proxy", "CR Proxy", "MRLTV Proxy", "Recommendation"
        ]
        out_show = out[show_cols].sort_values("Composite Score", ascending=False)

        render_formatted_table(out_show, use_container_width=True)

        st.markdown("**🏷️ Strategy-First Tiers: 1 strategy per tier + Growth (High/Low) + Intent (High/Low)**")
        tier_summary, tier_detail = build_strategy_tiers(out)
        render_formatted_table(tier_summary, use_container_width=True)

        st.markdown("**🔎 Sub-tier Details (state + sub channel rows)**")
        render_formatted_table(tier_detail, use_container_width=True)

        st.markdown("**🏁 Data-Performance Tiers: 1 performance group + Growth (High/Low) + Intent (High/Low)**")
        perf_tier_summary, perf_tier_detail = build_performance_tiers(out)
        render_formatted_table(perf_tier_summary, use_container_width=True)

        st.markdown("**🔎 Data-Performance Sub-tier Details (state + sub channel rows)**")
        render_formatted_table(perf_tier_detail, use_container_width=True)

        csv_bytes = out_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered channel-state table",
            data=csv_bytes,
            file_name="channel_group_state_recommendations.csv",
            mime="text/csv",
        )

    with tabs[4]:
        st.subheader("🌌 Neon Insights Cockpit")
        st.caption("Futuristic overview of growth, intent, performance, and strategy using current model outputs.")
        if fast_mode:
            st.info("Fast interaction mode is ON. Turn it OFF in the sidebar to render this heavy visual cockpit.")
        else:
            map_dfn = state_df.merge(state_extra_df, on="State", how="left")
            map_dfn["Expected_Additional_Clicks"] = map_dfn["Expected_Additional_Clicks"].fillna(0)
            map_dfn["Expected_Additional_Binds"] = map_dfn["Expected_Additional_Binds"].fillna(0)
            map_dfn["ROE Display"] = map_dfn["ROE"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
            map_dfn["CR Display"] = map_dfn["Combined Ratio"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
            map_dfn["LTV Display"] = map_dfn["Avg. MRLTV"].map(lambda x: "n/a" if pd.isna(x) else f"${x:,.0f}")
            map_dfn["Binds Display"] = map_dfn["Binds"].map(lambda x: "n/a" if pd.isna(x) else f"{x:,.0f}")
            map_dfn["Add Clicks Display"] = map_dfn["Expected_Additional_Clicks"].map(lambda x: f"{x:,.0f}")
            map_dfn["Add Binds Display"] = map_dfn["Expected_Additional_Binds"].map(lambda x: f"{x:,.1f}")
            map_dfn["Conflict Perf Label"] = (
                map_dfn["Performance Tone"].fillna("Unknown").astype(str)
                + " | "
                + map_dfn["Conflict Level"].fillna("Unknown").astype(str)
            )
            map_dfn.loc[~map_dfn["Conflict Perf Label"].isin(CONFLICT_PERF_COLOR.keys()), "Conflict Perf Label"] = "Unknown | Unknown"

            neon_mode = st.radio(
                "Map color mode",
                options=["Product Strategy", "Performance Group", "Conflict Highlight"],
                horizontal=True,
                key="tab4_map_mode",
            )
            if neon_mode == "Product Strategy":
                neon_color_col, neon_color_map = "Strategy Bucket", STRATEGY_COLOR
            elif neon_mode == "Performance Group":
                neon_color_col, neon_color_map = "ROE Performance Group", PERFORMANCE_GROUP_COLOR
            else:
                neon_color_col, neon_color_map = "Conflict Perf Label", CONFLICT_PERF_COLOR

            left, center, right = st.columns([1.1, 3.4, 1.1])
            with left:
                src = rec_df.groupby("Channel Groups", as_index=False)["Clicks"].sum().sort_values("Clicks", ascending=False)
                if not src.empty:
                    top_src = src.head(5).copy()
                    other = src["Clicks"].sum() - top_src["Clicks"].sum()
                    if other > 0:
                        top_src = pd.concat([top_src, pd.DataFrame([{"Channel Groups": "Other", "Clicks": other}])], ignore_index=True)
                    fig_src = px.pie(top_src, names="Channel Groups", values="Clicks", hole=0.65, title="Top Data Sources", template=plotly_template)
                    fig_src.update_layout(margin=dict(l=0, r=0, t=35, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbeafe"))
                    st.plotly_chart(fig_src, use_container_width=True, key="tab4_src")

            with center:
                fig_neon = px.choropleth(
                    map_dfn,
                    locations="State",
                    locationmode="USA-states",
                    scope="usa",
                    color=neon_color_col,
                    color_discrete_map=neon_color_map,
                    custom_data=["ROE Display", "CR Display", "LTV Display", "Binds Display", "Add Clicks Display", "Add Binds Display"],
                    template=plotly_template,
                )
                fig_neon.update_traces(
                    marker_line_color="#38bdf8",
                    marker_line_width=0.5,
                    hovertemplate=(
                        "<b>%{location}</b><br>"
                        "ROE %{customdata[0]} | CR %{customdata[1]}<br>"
                        "LTV %{customdata[2]} | Binds %{customdata[3]}<br>"
                        "Add Clicks %{customdata[4]} | Add Binds %{customdata[5]}<extra></extra>"
                    ),
                )
                fig_neon.update_geos(bgcolor="rgba(0,0,0,0)")
                fig_neon.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#dbeafe"),
                    height=760,
                )
                st.plotly_chart(fig_neon, use_container_width=True, key="tab4_map")

            with right:
                seg = rec_df.groupby("Segment", as_index=False)["Clicks"].sum().sort_values("Clicks", ascending=False)
                fig_seg = px.treemap(seg, path=["Segment"], values="Clicks", title="Segments", template=plotly_template)
                fig_seg.update_layout(margin=dict(l=0, r=0, t=35, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbeafe"))
                st.plotly_chart(fig_seg, use_container_width=True, key="tab4_seg")


if __name__ == "__main__":
    main()
