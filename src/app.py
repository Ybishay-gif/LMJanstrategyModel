import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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


def apply_price_effects(
    rec: pd.DataFrame, price_eval_df: pd.DataFrame
) -> pd.DataFrame:
    effects = (
        price_eval_df[["Channel Groups", "Price Adjustment Percent", "Clicks Lift %", "CPC Lift %"]]
        .groupby(["Channel Groups", "Price Adjustment Percent"], as_index=False)
        .mean(numeric_only=True)
    )
    effect_dict: dict[str, pd.DataFrame] = {
        ch: g.sort_values("Price Adjustment Percent")
        for ch, g in effects.groupby("Channel Groups")
    }

    def lookup(row: pd.Series) -> pd.Series:
        g = effect_dict.get(row["Channel Groups"])
        if g is None or g.empty:
            return pd.Series([row["Suggested Price Adjustment %"], 0.0, 0.0])
        target = row["Suggested Price Adjustment %"]
        idx = (g["Price Adjustment Percent"] - target).abs().idxmin()
        near = g.loc[idx]
        return pd.Series([near["Price Adjustment Percent"], near["Clicks Lift %"], near["CPC Lift %"]])

    mapped = rec.apply(lookup, axis=1)
    mapped.columns = ["Applied Price Adjustment %", "Clicks Lift %", "CPC Lift %"]
    out = rec.copy()
    out[["Applied Price Adjustment %", "Clicks Lift %", "CPC Lift %"]] = mapped
    return out


def prepare_state(state_df: pd.DataFrame, strategy_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(state_df)
    for col in ["ROE", "Combined Ratio", "Performance", "CPB", "Target CPB", "Clicks", "Binds", "Avg. MRLTV"]:
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
    return out


def prepare_state_seg(state_seg_df: pd.DataFrame, state_df: pd.DataFrame) -> pd.DataFrame:
    seg = normalize_columns(state_seg_df).copy()
    st = normalize_columns(state_df).copy()

    for col in ["ROE", "Combined Ratio", "Performance", "CPB", "Target CPB", "Clicks", "Binds", "Avg. MRLTV"]:
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
    return out


def prepare_channel_state(channel_state_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(channel_state_df).copy()
    for col in [
        "Bids", "Avg. CPC", "Avg. Bid", "Impressions", "SOV", "Clicks", "Bids to Clicks",
        "ROE", "Combined Ratio", "Performance", "CPB", "Target CPB", "Quote Start Rate",
        "Clicks to Quotes", "Avg. MRLTV", "Binds"
    ]:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    df["Segment"] = df["Channel Groups"].map(extract_segment)
    return df


def prepare_price_exploration(price_df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(price_df).copy()
    for col in ["Price Adjustment Percent", "Bids", "Avg. CPC", "Avg. Bid", "Impressions", "SOV", "Clicks"]:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    df["Win Rate"] = np.where(df["Bids"] > 0, df["Clicks"] / df["Bids"], np.nan)

    base = df[df["Price Adjustment Percent"] == 0][
        ["Channel Groups", "Clicks", "Avg. CPC", "Win Rate", "SOV"]
    ].rename(
        columns={
            "Clicks": "Baseline Clicks",
            "Avg. CPC": "Baseline CPC",
            "Win Rate": "Baseline Win Rate",
            "SOV": "Baseline SOV",
        }
    )

    out = df.merge(base, on="Channel Groups", how="left")
    out["Clicks Lift %"] = np.where(out["Baseline Clicks"] > 0, out["Clicks"] / out["Baseline Clicks"] - 1, np.nan)
    out["CPC Lift %"] = np.where(out["Baseline CPC"] > 0, out["Avg. CPC"] / out["Baseline CPC"] - 1, np.nan)
    out["Win Rate Lift %"] = np.where(out["Baseline Win Rate"] > 0, out["Win Rate"] / out["Baseline Win Rate"] - 1, np.nan)

    out["Growth Opportunity Score"] = (
        0.50 * out["Clicks Lift %"].fillna(0)
        + 0.30 * out["Win Rate Lift %"].fillna(0)
        + 0.20 * (1 - out["Baseline SOV"].fillna(0.5))
        - settings.cpc_penalty_weight * np.maximum(out["CPC Lift %"].fillna(0), 0)
    )

    feasible = out[out["CPC Lift %"].fillna(0) <= settings.max_cpc_increase_pct / 100.0].copy()
    best = feasible.sort_values("Growth Opportunity Score", ascending=False).groupby("Channel Groups", as_index=False).first()

    keep_cols = [
        "Channel Groups", "Price Adjustment Percent", "Growth Opportunity Score",
        "Clicks Lift %", "CPC Lift %", "Win Rate Lift %"
    ]
    return out, best[keep_cols]


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
    rec["Growth Score"] = rec["Growth Opportunity Score"].fillna(0)

    rec["Composite Score"] = (
        settings.growth_weight * rec["Growth Score"] + settings.profit_weight * rec["Performance Score"]
    ) * rec["Strategy Scale"].fillna(0.5)

    rec.loc[
        (rec["Composite Score"] >= settings.controlled_cutoff)
        & (rec["Composite Score"] < settings.aggressive_cutoff),
        "Suggested Price Adjustment %",
    ] = np.minimum(rec["Suggested Price Adjustment %"], 10)
    rec.loc[
        (rec["Composite Score"] >= settings.maintain_cutoff)
        & (rec["Composite Score"] < settings.controlled_cutoff),
        "Suggested Price Adjustment %",
    ] = 0
    rec.loc[rec["Composite Score"] < settings.maintain_cutoff, "Suggested Price Adjustment %"] = -5

    rec["Strategy Max Adj %"] = rec["Strategy Bucket"].apply(lambda x: strategy_max_adjustment(x, settings))
    rec["Suggested Price Adjustment %"] = np.minimum(rec["Suggested Price Adjustment %"], rec["Strategy Max Adj %"])

    rec["Intent Score"] = (
        0.60 * rec["Quote Start Rate"].fillna(0)
        + 0.40 * rec["Clicks to Quotes"].fillna(0)
    )
    rec.loc[rec["Intent Score"] < settings.min_intent_for_scale, "Suggested Price Adjustment %"] = np.minimum(
        rec["Suggested Price Adjustment %"], 0
    )
    hard_pullback = (rec["ROE Proxy"] < settings.roe_pullback_floor) | (rec["CR Proxy"] > settings.cr_pullback_ceiling)
    rec.loc[hard_pullback, "Suggested Price Adjustment %"] = -10

    rec = apply_price_effects(rec, price_eval_df)
    rec["Clicks Lift %"] = rec["Clicks Lift %"].fillna(0)
    rec["CPC Lift %"] = rec["CPC Lift %"].fillna(0)

    rec["Expected Additional Clicks"] = rec["Clicks"] * rec["Clicks Lift %"]
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


def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_cols = {
        "ROE", "Combined Ratio", "Performance",
        "ROE Proxy", "CR Proxy", "Performance Score",
        "Clicks to Binds", "Seg Clicks to Binds", "Clicks to Binds Proxy",
    }
    currency_cols = {
        "Avg. MRLTV", "State Avg. MRLTV", "Seg Avg. MRLTV", "MRLTV Proxy", "Avg_LTV", "Avg_MRLTV",
        "CPB", "State CPB", "Target CPB", "Avg. CPC", "Avg. Bid", "Baseline CPC", "Expected Additional Cost",
    }

    for c in out.columns:
        if c in pct_cols and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt_pct)
        elif c in currency_cols and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt_cur)
        elif c in {"Suggested Price Adjustment %", "Applied Price Adjustment %", "Suggested_Price_Adjustment_pct"} and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.0f}%")
    return out


def build_tier_table_strategy(rec: pd.DataFrame) -> pd.DataFrame:
    tier = rec.copy()
    tier["Growth Tier"] = quantile_bucket(tier["Growth Score"].fillna(0), ["Low", "Mid", "High"])
    tier["Intent Tier"] = quantile_bucket(tier["Intent Score"].fillna(0), ["Low", "Mid", "High"])
    tier["Strategy Tier"] = tier["Strategy Bucket"].fillna("Unknown")

    out = tier.groupby(["Growth Tier", "Intent Tier", "Strategy Tier"], as_index=False).agg(
        States=("State", lambda x: ", ".join(sorted(set(x)))),
        Channel_Groups=("Channel Groups", lambda x: ", ".join(sorted(set(x)))),
        Rows=("Channel Groups", "count"),
        Additional_Clicks=("Expected Additional Clicks", "sum"),
        Additional_Binds=("Expected Additional Binds", "sum"),
    )
    out = out.sort_values("Additional_Clicks", ascending=False).head(10)
    out.insert(0, "Tier", [f"Tier {i}" for i in range(1, len(out) + 1)])
    return out


def build_tier_table_perf(rec: pd.DataFrame) -> pd.DataFrame:
    tier = rec.copy()
    tier["Growth Tier"] = quantile_bucket(tier["Growth Score"].fillna(0), ["Low", "Mid", "High"])
    tier["Intent Tier"] = quantile_bucket(tier["Intent Score"].fillna(0), ["Low", "Mid", "High"])
    tier["Performance Tier"] = quantile_bucket(tier["Performance Score"].fillna(0), ["Low", "Mid", "High", "Very High"])

    out = tier.groupby(["Growth Tier", "Intent Tier", "Performance Tier"], as_index=False).agg(
        States=("State", lambda x: ", ".join(sorted(set(x)))),
        Channel_Groups=("Channel Groups", lambda x: ", ".join(sorted(set(x)))),
        Rows=("Channel Groups", "count"),
        Additional_Clicks=("Expected Additional Clicks", "sum"),
        Additional_Binds=("Expected Additional Binds", "sum"),
    )
    out = out.sort_values("Additional_Clicks", ascending=False).head(10)
    out.insert(0, "Tier", [f"Tier {i}" for i in range(1, len(out) + 1)])
    return out


def main() -> None:
    with st.sidebar:
        dark_mode = st.toggle("Dark mode", value=True)
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
        st.markdown("**Scoring Weights**")
        growth_weight = st.slider("Growth weight", 0.0, 1.0, 0.55, 0.05)
        profit_weight = st.slider("Profitability weight", 0.0, 1.0, 0.45, 0.05)

        st.markdown("**Guardrails**")
        max_cpc_increase_pct = st.slider("Max CPC increase %", 0, 40, 12, 1)
        min_bids_channel_state = st.slider("Min bids for reliable channel-state", 1, 20, 5, 1)
        cpc_penalty_weight = st.slider("CPC penalty", 0.0, 1.5, 0.65, 0.05)
        min_intent_for_scale = st.slider("Min intent to allow positive scaling", 0.0, 1.0, 0.92, 0.01)
        roe_pullback_floor = st.slider("ROE pullback floor", -1.0, 0.5, -0.20, 0.01)
        cr_pullback_ceiling = st.slider("Combined ratio pullback ceiling", 0.8, 1.5, 1.15, 0.01)

        st.markdown("**Score Cutoffs**")
        aggressive_cutoff = st.slider("Aggressive cutoff", 0.3, 1.0, 0.65, 0.01)
        controlled_cutoff = st.slider("Controlled cutoff", 0.2, aggressive_cutoff, min(0.45, aggressive_cutoff), 0.01)
        maintain_cutoff = st.slider("Maintain cutoff", 0.0, controlled_cutoff, min(0.30, controlled_cutoff), 0.01)

        st.markdown("**Strategy Max Adjustment (%)**")
        max_adj_strongest = st.slider("Strongest Momentum cap", -10, 40, 20, 1)
        max_adj_moderate = st.slider("Moderate Momentum cap", -10, 30, 12, 1)
        max_adj_minimal = st.slider("Minimal Growth cap", -10, 20, 5, 1)
        max_adj_constrained = st.slider("Constrained / Inactive cap", -10, 15, 0, 1)

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

    state_df = prepare_state(state_raw, strategy_df)
    state_seg_df = prepare_state_seg(state_seg_raw, state_raw)
    channel_state_df = prepare_channel_state(channel_state_raw)
    price_eval, best_adj = prepare_price_exploration(price_raw, settings)
    rec_df, state_extra_df, state_seg_extra_df, channel_summary_df = build_model_tables(
        state_df, state_seg_df, channel_state_df, best_adj, price_eval, settings
    )

    tabs = st.tabs(["🗺️ Tab 1: State Momentum Map", "📊 Tab 2: Channel Group Analysis", "🧠 Tab 3: Channel Group and States"])

    with tabs[0]:
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

        fig = px.choropleth(
            map_df,
            locations="State",
            locationmode="USA-states",
            scope="usa",
            color="Strategy Bucket",
            color_discrete_map=STRATEGY_COLOR,
            title="US Map: Strategy Bucket + State KPIs",
        )
        fig.update_traces(
            customdata=np.stack(
                [
                    map_df["Strategy Bucket"].astype(str),
                    map_df["Indicator"].astype(str),
                    map_df["Conflict Label"].astype(str),
                    map_df["ROE Display"].astype(str),
                    map_df["CR Display"].astype(str),
                    map_df["Perf Display"].astype(str),
                    map_df["Binds Display"].astype(str),
                    map_df["LTV Display"].astype(str),
                    map_df["Add Clicks Display"].astype(str),
                    map_df["Add Binds Display"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b style='font-size:15px;'>%{location}</b><br>"
                "<span style='opacity:0.88;'>%{customdata[0]}</span><br>"
                "<span style='opacity:0.78;'>%{customdata[1]} %{customdata[2]}</span>"
                "<br>━━━━━━━━━━━━<br>"
                "<b>ROE</b> %{customdata[3]}  ·  <b>CR</b> %{customdata[4]}<br>"
                "<b>Perf</b> %{customdata[5]}  ·  <b>Binds</b> %{customdata[6]}<br>"
                "<b>Avg LTV</b> %{customdata[7]}<br>"
                "<br><b>Growth Upside</b><br>"
                "Additional Clicks: <b>%{customdata[8]}</b><br>"
                "Additional Binds: <b>%{customdata[9]}</b><extra></extra>"
            ),
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            legend_title_text="Strategy",
            template=plotly_template,
            clickmode="event+select",
            hoverlabel=dict(
                bgcolor="#0F172A" if dark_mode else "#F8FAFC",
                bordercolor="#334155" if dark_mode else "#CBD5E1",
                font=dict(color="#E2E8F0" if dark_mode else "#0F172A", size=13),
                align="left",
            ),
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

        if selected_state:
            row = map_df[map_df["State"] == selected_state].head(1)
            if row.empty:
                st.warning("No state-level data found for the selected state.")
                return
            seg_view = state_seg_df[state_seg_df["State"] == selected_state].merge(
                state_seg_extra_df[state_seg_extra_df["State"] == selected_state],
                on=["State", "Segment"],
                how="left",
            )
            seg_view["Expected_Additional_Clicks"] = seg_view["Expected_Additional_Clicks"].fillna(0)
            seg_view["Expected_Additional_Binds"] = seg_view["Expected_Additional_Binds"].fillna(0)

            with st.container(border=True):
                st.subheader(f"🔎 State Deep Dive: {selected_state}")
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

                c5, c6 = st.columns(2)
                c5.metric("✨ State Additional Clicks", f"{row['Expected_Additional_Clicks'].iloc[0]:,.0f}")
                c6.metric("🎉 State Additional Binds", f"{row['Expected_Additional_Binds'].iloc[0]:,.1f}")

                st.markdown("**🧩 Per-Segment KPI + Opportunity**")
                seg_show = seg_view[[
                    "Segment", "Clicks", "Binds", "Clicks to Binds", "ROE", "Combined Ratio", "Avg. MRLTV",
                    "Expected_Additional_Clicks", "Expected_Additional_Binds"
                ]].sort_values("Expected_Additional_Clicks", ascending=False)
                st.dataframe(
                    format_display_df(seg_show),
                    use_container_width=True,
                )

        st.markdown("**State Strategy vs Actual Indicator**")
        indicator_view = map_df[[
            "State", "Strategy Bucket", "Conflict Arrow", "Conflict Level", "Performance Tone", "ROE", "Combined Ratio", "Performance"
        ]].sort_values(["Conflict Level", "State"])
        indicator_view["Indicator"] = np.where(
            indicator_view["Performance Tone"] == "Good",
            "🟢",
            np.where(indicator_view["Performance Tone"] == "Poor", "🔴", "🟡"),
        )
        indicator_view["Match"] = indicator_view["Indicator"] + " " + indicator_view["Conflict Arrow"] + " " + indicator_view["Conflict Level"]
        st.dataframe(
            format_display_df(indicator_view[["State", "Strategy Bucket", "Match", "ROE", "Combined Ratio", "Performance"]]),
            use_container_width=True,
        )

    with tabs[1]:
        st.subheader("📊 Channel Group Analysis")

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

        grp = filt.groupby(["Channel Groups", "Segment"], as_index=False).agg(
            Clicks=("Clicks", "sum"),
            Suggested_Price_Adjustment_pct=("Suggested Price Adjustment %", "median"),
            Applied_Price_Adjustment_pct=("Applied Price Adjustment %", "median"),
            Additional_Clicks=("Expected Additional Clicks", "sum"),
            Additional_Binds=("Expected Additional Binds", "sum"),
            ROE=("ROE Proxy", "mean"),
            Combined_Ratio=("CR Proxy", "mean"),
            Avg_LTV=("MRLTV Proxy", "mean"),
        ).sort_values("Additional_Clicks", ascending=False)

        st.dataframe(format_display_df(grp), use_container_width=True)

    with tabs[2]:
        st.subheader("🧠 Channel Group + State Recommendations")

        c1, c2, c3, c4, c5 = st.columns(5)
        states = sorted(rec_df["State"].dropna().unique().tolist())
        strategies = sorted(rec_df["Strategy Bucket"].dropna().unique().tolist())
        segments = sorted(rec_df["Segment"].dropna().unique().tolist())
        channels = sorted(rec_df["Channel Groups"].dropna().unique().tolist())

        fs = c1.multiselect("State", options=states, default=states, key="tab3_state")
        fst = c2.multiselect("State Strategy", options=strategies, default=strategies, key="tab3_strategy")
        fseg = c3.multiselect("Segment", options=segments, default=segments, key="tab3_segment")
        fch = c4.multiselect("Channel Group", options=channels, default=channels[: min(len(channels), 30)], key="tab3_channel")
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
            "Expected Additional Clicks", "Expected Additional Binds", "Expected Additional Cost", "Growth Score",
            "Performance Score", "Composite Score", "ROE Proxy", "CR Proxy", "MRLTV Proxy", "Recommendation"
        ]
        out_show = out[show_cols].sort_values("Composite Score", ascending=False)

        st.dataframe(
            styled_table(
                format_display_df(out_show),
                perf_cols=["Performance Score", "ROE Proxy", "CR Proxy", "MRLTV Proxy"],
                strategy_cols=["Strategy Bucket", "Composite Score", "Recommendation"],
            ),
            use_container_width=True,
        )

        st.markdown("**🏷️ Tiers by Growth + Intent + Product Strategy (Top 10)**")
        tier_strategy = build_tier_table_strategy(out)
        st.dataframe(format_display_df(tier_strategy), use_container_width=True)

        st.markdown("**🏁 Tiers by Growth + Intent + Performance Score (Top 10)**")
        tier_perf = build_tier_table_perf(out)
        st.dataframe(format_display_df(tier_perf), use_container_width=True)

        csv_bytes = out_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered channel-state table",
            data=csv_bytes,
            file_name="channel_group_state_recommendations.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
