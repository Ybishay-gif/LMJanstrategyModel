import re
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

st.set_page_config(page_title="Insurance Growth Navigator", layout="wide")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ADMIN_EMAIL,
    CONFLICT_PERF_COLOR,
    DARK_CSS,
    DEFAULT_PATHS,
    LIGHT_CSS,
    OPTIMIZATION_MODES,
    OVERRIDES_PATH,
    PERFORMANCE_GROUP_COLOR,
    STATE_CENTER,
    STRATEGY_COLOR,
    STRATEGY_SCALE,
    Settings,
    classify_perf_group,
    effective_cpc_cap_pct,
    effective_cpc_penalty,
    mode_factor,
)
from ui_utils import styled_table, _safe_weighted_mean, format_display_df, render_formatted_table
from auth_layer import (
    normalize_email,
    load_allowed_emails,
    save_allowed_emails,
    now_iso,
    load_auth_users,
    save_auth_users,
    hash_password,
    verify_password,
    make_invite_token,
    make_session_token,
    build_invite_link,
    issue_session,
    resolve_session_token,
    send_invite_email,
    resolve_invite_token,
    render_auth_gate,
    render_settings_panel,
    qp_value,
    build_query_url,
    perform_logout,
    render_top_icons,
)
from data_io import read_csv, read_state_strategy, parse_state_strategy_text, file_mtime
from core_helpers import (
    to_numeric,
    normalize_columns,
    extract_segment,
    quantile_bucket,
    format_adj_option_label,
    parse_adj_from_label,
    apply_grid_preset,
    tab5_grid_component_key,
    as_float,
    close_adj,
)
from storage_layer import (
    load_overrides_from_disk,
    save_overrides_to_disk,
    load_analytics_presets,
    save_analytics_presets,
)
from analytics_builders import (
    build_price_exploration_master_detail,
    build_price_exploration_detail_lookup,
    build_general_analytics_df,
)


@st.cache_data(show_spinner=False, hash_funcs={Settings: lambda s: tuple(vars(s).items())})
def build_all_from_paths(
    strategy_path: str,
    state_path: str,
    state_seg_path: str,
    channel_group_path: str,
    price_path: str,
    channel_state_path: str,
    settings: Settings,
    mtime_signature: tuple[float, float, float, float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # mtime_signature is only for cache invalidation when files change.
    _ = mtime_signature
    strategy_df = read_state_strategy(strategy_path)
    state_raw = read_csv(state_path)
    state_seg_raw = read_csv(state_seg_path)
    _ = read_csv(channel_group_path)
    price_raw = read_csv(price_path)
    channel_state_raw = read_csv(channel_state_path)

    state_df = prepare_state(state_raw, strategy_df, settings)
    state_seg_df = prepare_state_seg(state_seg_raw, state_raw, settings)
    channel_state_df = prepare_channel_state(channel_state_raw)
    price_eval, best_adj = prepare_price_exploration(price_raw, settings)
    rec_df, state_extra_df, state_seg_extra_df, channel_summary_df = build_model_tables(
        state_df, state_seg_df, channel_state_df, best_adj, price_eval, settings
    )
    return rec_df, state_df, state_seg_df, price_eval, state_extra_df, state_seg_extra_df


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


def strategy_min_adjustment(bucket: str, settings: Settings) -> float:
    f = mode_factor(settings.optimization_mode)
    # Growth-focused modes do not cut bids by default.
    if f >= 0.75:
        return 0.0
    # Balanced allows mild pullback in weak strategy states.
    if f >= 0.5:
        if bucket == "Strongest Momentum":
            return 0.0
        if bucket == "Moderate Momentum":
            return -5.0
        if bucket == "Minimal Growth":
            return -10.0
        return -15.0
    # Cost-leaning modes allow broader pullback while respecting strategy.
    if bucket == "Strongest Momentum":
        return -5.0
    if bucket == "Moderate Momentum":
        return -10.0
    if bucket == "Minimal Growth":
        return -15.0
    return -25.0


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

    state_curves: dict[tuple[str, str], pd.DataFrame] = {}
    if "State" in px.columns:
        state_curves = {
            (str(st), str(ch)): g.sort_values("Price Adjustment Percent")
            for (st, ch), g in px.groupby(["State", "Channel Groups"])
        }
    ch_curves = {str(ch): g.sort_values("Price Adjustment Percent") for ch, g in px.groupby("Channel Groups")}

    def _pick_from_curve(curve: pd.DataFrame, row: pd.Series) -> Optional[pd.Series]:
        if curve is None or curve.empty:
            return None
        curve = curve.copy()
        bucket = str(row.get("Strategy Bucket", "") or "")
        min_adj = strategy_min_adjustment(bucket, settings)
        max_adj = strategy_max_adjustment(bucket, settings)
        curve = curve[
            (pd.to_numeric(curve["Price Adjustment Percent"], errors="coerce").fillna(0.0) >= min_adj)
            & (pd.to_numeric(curve["Price Adjustment Percent"], errors="coerce").fillna(0.0) <= max_adj)
        ]
        if curve.empty:
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

        f = mode_factor(settings.optimization_mode)
        # Strategy-tuned tradeoff: momentum states bias to growth; constrained states bias to CPC/perf.
        wr_w = 1.0 + 0.8 * f
        cpc_w = 0.55 + 1.20 * (1 - f)
        perf_w = 0.30 + 0.90 * (1 - f)
        save_w = 0.20 + 0.70 * (1 - f)
        if bucket == "Strongest Momentum":
            wr_w *= 1.35
            cpc_w *= 0.70
        elif bucket == "Moderate Momentum":
            wr_w *= 1.15
            cpc_w *= 0.90
        elif bucket == "Minimal Growth":
            wr_w *= 0.80
            cpc_w *= 1.30
        elif bucket in {"LTV Constrained", "Closure Constrained", "Inactive/Low Spend"}:
            wr_w *= 0.65
            cpc_w *= 1.55
            perf_w *= 1.20

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
            perf_gain = 0.0
            if pd.notna(actual_perf) and pd.notna(new_perf):
                perf_drop = max(actual_perf - new_perf, 0.0)
                perf_gain = new_perf - actual_perf
            perf_ok = True
            if pd.notna(new_perf):
                perf_ok = (new_perf >= settings.min_new_performance) and (perf_drop <= settings.max_perf_drop)
            adj = float(c.get("Price Adjustment Percent", 0) or 0)
            utility = (
                wr_w * wr_lift
                - cpc_w * max(cpc_lift, 0.0)
                + save_w * max(-cpc_lift, 0.0)
                + perf_w * perf_gain
            )
            if f >= 0.75 and adj < 0:
                utility -= 2.0
            if f <= 0.50 and adj < 0:
                utility += 0.02 * abs(adj)
            if f <= 0.25 and bucket == "Strongest Momentum" and adj < 0:
                utility -= 0.4
            scored.append(
                {
                    "cand": c,
                    "add_binds": add_binds,
                    "add_clicks": add_clicks,
                    "perf_drop": perf_drop,
                    "perf_gain": perf_gain,
                    "wr_lift": wr_lift,
                    "cpc_lift": cpc_lift,
                    "adj": adj,
                    "utility": utility,
                    "new_perf": new_perf,
                    "perf_ok": perf_ok,
                }
            )
        if not scored:
            return None
        valid = [x for x in scored if x["perf_ok"]]
        pool = valid if valid else scored
        if pool:
            if settings.optimization_mode == "Max Growth" and bucket == "Strongest Momentum":
                best = sorted(
                    pool,
                    key=lambda x: (x["wr_lift"], x["add_binds"], -x["cpc_lift"], x["adj"]),
                    reverse=True,
                )[0]
                return best["cand"]
            best = sorted(
                pool,
                key=lambda x: (x["utility"], x["wr_lift"], x["add_binds"], -x["cpc_lift"]),
                reverse=True,
            )[0]
            return best["cand"]
        # If all points hurt performance too much, choose the most conservative feasible degradation.
        best = sorted(
            scored,
            key=lambda x: (x["perf_drop"], -x["utility"], -x["add_binds"], x["adj"]),
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
    rec["Applied Price Adjustment %"] = rec["Suggested Price Adjustment %"]
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

    rec["Strategy Max Adj %"] = rec["Strategy Bucket"].apply(lambda x: strategy_max_adjustment(x, settings))
    rec["Suggested Price Adjustment %"] = np.minimum(rec["Suggested Price Adjustment %"], rec["Strategy Max Adj %"])
    rec["Applied Price Adjustment %"] = np.minimum(rec["Applied Price Adjustment %"], rec["Strategy Max Adj %"])
    rec["Clicks Lift %"] = rec["Clicks Lift %"].fillna(0)
    rec["Win Rate Lift %"] = rec["Win Rate Lift %"].fillna(0)
    rec["CPC Lift %"] = rec["CPC Lift %"].fillna(0)
    rec.loc[~rec["Has Sig Price Evidence"], ["Suggested Price Adjustment %", "Applied Price Adjustment %", "Clicks Lift %", "Win Rate Lift %", "CPC Lift %"]] = 0.0
    # Growth is based on win-rate uplift applied to the row's bid volume.
    rec["Lift Proxy %"] = rec["Win Rate Lift %"]
    no_growth = rec["Win Rate Lift %"] < 0
    allow_negative_moves = mode_factor(settings.optimization_mode) <= 0.5
    if not allow_negative_moves:
        rec.loc[no_growth, "Suggested Price Adjustment %"] = 0
        rec.loc[no_growth, "Applied Price Adjustment %"] = 0
        rec.loc[no_growth, "Clicks Lift %"] = 0
        rec.loc[no_growth, "Win Rate Lift %"] = 0
        rec.loc[no_growth, "Lift Proxy %"] = 0
        rec.loc[no_growth, "CPC Lift %"] = 0
    else:
        # In balanced/cost modes keep negative win-rate moves only if they reduce CPC.
        economically_bad = no_growth & (rec["CPC Lift %"] >= 0)
        rec.loc[economically_bad, "Suggested Price Adjustment %"] = 0
        rec.loc[economically_bad, "Applied Price Adjustment %"] = 0
        rec.loc[economically_bad, "Clicks Lift %"] = 0
        rec.loc[economically_bad, "Win Rate Lift %"] = 0
        rec.loc[economically_bad, "Lift Proxy %"] = 0
        rec.loc[economically_bad, "CPC Lift %"] = 0

    fallback_win_rate = pd.Series(np.where(rec["Bids"] > 0, rec["Clicks"] / rec["Bids"], 0), index=rec.index)
    current_win_rate = rec["Bids to Clicks"].combine_first(fallback_win_rate)
    rec["Test-based Additional Clicks"] = np.where(
        rec["Has Sig Price Evidence"],
        rec["Bids"].fillna(0) * current_win_rate.fillna(0) * rec["Win Rate Lift %"],
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
    min_adj = -30.0 if mode_factor(settings.optimization_mode) <= 0.5 else 0.0
    p = p[p["Price Adjustment Percent"].fillna(0) >= min_adj].copy()
    p = p[p["Stat Sig Price Point"] == True].copy()
    p = p[p["CPC Lift %"].fillna(0) <= effective_cpc_cap_pct(settings) / 100.0].copy()
    if p.empty:
        return pd.DataFrame(), "None"

    src = "Channel"
    if "State" in p.columns:
        ps = p[p["State"] == state].copy()
        pch = (
            p.groupby(["Price Adjustment Percent"], as_index=False)
            .agg(
                Bids=("Bids", "sum"),
                Clicks=("Clicks", "sum"),
                **{"Win Rate Lift %": ("Win Rate Lift %", "mean")},
                **{"CPC Lift %": ("CPC Lift %", "mean")},
            )
            .sort_values("Price Adjustment Percent")
        )
        if not ps.empty:
            ps = ps.sort_values("Price Adjustment Percent").copy()
            ps["Source Used"] = "State+Channel"
            pch["Source Used"] = "Channel Fallback"
            ps_adj = set(pd.to_numeric(ps["Price Adjustment Percent"], errors="coerce").dropna().astype(float).tolist())
            pch_missing = pch[~pd.to_numeric(pch["Price Adjustment Percent"], errors="coerce").astype(float).isin(ps_adj)].copy()
            p = pd.concat([ps, pch_missing], ignore_index=True)
            src = "Mixed" if not pch_missing.empty else "State+Channel"
        else:
            p = pch
            p["Source Used"] = "Channel Fallback"
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
    if "Source Used" not in p.columns:
        p["Source Used"] = src
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
    allow_negative_moves = mode_factor(settings.optimization_mode) <= 0.5
    for _, cr in candidates.sort_values("Price Adjustment Percent").iterrows():
        wr_l = float(cr.get("Win Rate Lift %", 0) or 0)
        cpc_l = float(cr.get("CPC Lift %", 0) or 0)
        wr_effect = wr_l if allow_negative_moves else max(wr_l, 0)
        add_clicks = float((rsel["Bids"].fillna(0) * base_wr * wr_effect).sum())
        add_binds = float((rsel["Bids"].fillna(0) * base_wr * wr_effect * c2b).sum())
        exp_cost = float(((rsel["Clicks"] + (rsel["Bids"] * base_wr * wr_effect)) * rsel["Avg. CPC"] * (1 + cpc_l)).sum())
        new_cpb = (exp_cost / (cur_binds + add_binds)) if (cur_binds + add_binds) > 0 else np.nan
        cpb_impact = (new_cpb / cur_cpb - 1) if pd.notna(new_cpb) and pd.notna(cur_cpb) and cur_cpb > 0 else np.nan
        out_rows.append(
            {
                "Bid Adj %": float(cr["Price Adjustment Percent"]),
                "Sig Icon": cr.get("Sig Icon", "⚪"),
                "Sig Level": cr.get("Sig Level", "n/a"),
                "Source Used": str(cr.get("Source Used", source_used)),
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
    out = pd.concat(rows, ignore_index=True)
    if "Source Used" not in out.columns:
        out["Source Used"] = "Unknown"
    return out


@st.cache_data(show_spinner=False, hash_funcs={Settings: lambda s: tuple(vars(s).items())})
def nearest_available_adj(channel_group: str, target_adj: float, popup_state_df: pd.DataFrame) -> tuple[float, str]:
    if popup_state_df is None or popup_state_df.empty:
        return float(target_adj), "No test-point list found; used requested value."
    opts = popup_state_df.loc[
        popup_state_df["Channel Groups"].astype(str) == str(channel_group), "Bid Adj %"
    ]
    opts = pd.to_numeric(opts, errors="coerce").dropna().unique().tolist()
    if not opts:
        return float(target_adj), "No valid test points for this channel; used requested value."
    opts = sorted(float(x) for x in opts)
    mapped = min(opts, key=lambda x: (abs(x - float(target_adj)), x))
    if abs(mapped - float(target_adj)) > 1e-9:
        return mapped, f"Mapped to nearest valid test point ({mapped:+.0f}%)."
    return mapped, "Exact test point."


def _kpi_text(v) -> str:
    if isinstance(v, str):
        return v
    if v is None or pd.isna(v):
        return "n/a"
    return str(v)


def render_kpi_tiles(items: list[dict], cols: int = 4) -> None:
    if not items:
        return
    for i in range(0, len(items), cols):
        row = items[i:i + cols]
        c = st.columns(cols)
        for j in range(cols):
            if j >= len(row):
                continue
            it = row[j]
            label = _kpi_text(it.get("label", ""))
            value = _kpi_text(it.get("value", "n/a"))
            sub = _kpi_text(it.get("sub", ""))
            c[j].markdown(
                (
                    "<div class='kpi-tile'>"
                    f"<div class='kpi-label'>{label}</div>"
                    f"<div class='kpi-value'>{value}</div>"
                    f"<div class='kpi-sub'>{sub}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def fixed_height_container(height: int, key: str = "scroll_box"):
    try:
        return st.container(height=height, border=True, key=key)
    except TypeError:
        return st.container(border=True)


def to_bool(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float)) and not pd.isna(v):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off", ""}:
            return False
    return False


def confirm_action_modal(title: str, message: str, confirm_key: str, cancel_key: str) -> None:
    @st.dialog(title)
    def _modal():
        st.write(message)
        c1, c2 = st.columns(2)
        if c1.button("Confirm", key=confirm_key):
            st.session_state["tab1_modal_confirmed"] = True
            st.rerun()
        if c2.button("Cancel", key=cancel_key):
            st.session_state["tab1_modal_pending_action"] = ""
            st.session_state["tab1_modal_confirmed"] = False
            st.rerun()
    _modal()


def apply_user_bid_overrides(rec_df: pd.DataFrame, price_eval_df: pd.DataFrame, settings: Settings, overrides: dict) -> pd.DataFrame:
    if not overrides:
        return rec_df
    rec = rec_df.copy()
    for i, row in rec.iterrows():
        key = f"{row.get('State','')}|{row.get('Channel Groups','')}"
        o = overrides.get(key)
        if not isinstance(o, dict) or not o.get("apply", False):
            continue
        target = float(o.get("adj", row.get("Applied Price Adjustment %", 0.0)))
        cand, _ = build_adjustment_candidates(
            price_eval_df,
            str(row.get("State", "")),
            str(row.get("Channel Groups", "")),
            settings,
        )
        if cand is None or cand.empty:
            continue
        gg = cand.copy()
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
    if not render_auth_gate():
        return

    action = qp_value("action", "")
    if action == "logout":
        perform_logout()
        return

    if "global_optimization_mode" not in st.session_state:
        st.session_state["global_optimization_mode"] = "Max Growth"

    user_now = normalize_email(st.session_state.get("auth_user", ""))
    is_admin = user_now == ADMIN_EMAIL
    view = qp_value("view", "main").strip().lower() or "main"

    with st.sidebar:
        st.caption(f"Signed in as `{st.session_state.get('auth_user', '')}`")
        st.divider()
        dark_mode = st.toggle("Dark mode", value=True)
        fast_mode = st.toggle("Fast interaction mode", value=True, help="Reduces heavy chart rendering for faster clicks/saves.")
    st.markdown(DARK_CSS if dark_mode else LIGHT_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .kpi-tile {
            border: 1px solid rgba(148,163,184,0.26);
            border-radius: 12px;
            padding: 10px 12px;
            background: rgba(10,15,28,0.82);
            min-height: 84px;
            margin-top: 6px;
            margin-bottom: 8px;
            position: relative;
            z-index: 1;
        }
        .kpi-label {
            color: #93c5fd;
            font-size: 0.78rem;
            margin-bottom: 4px;
        }
        .kpi-value {
            color: #e2e8f0;
            font-weight: 700;
            font-size: 1.22rem;
            line-height: 1.15;
            margin-bottom: 2px;
        }
        .kpi-sub {
            color: #94a3b8;
            font-size: 0.72rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    plotly_template = "plotly_dark" if dark_mode else "plotly_white"

    render_top_icons(is_admin=is_admin, settings_view=(view == "settings"))
    if view == "settings":
        st.title("Settings")
        render_settings_panel()
        return

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
        optimization_mode = st.select_slider(
            "Recommendation Strategy",
            options=OPTIMIZATION_MODES,
            value=st.session_state.get("global_optimization_mode", "Max Growth"),
            key="global_optimization_mode",
        )
        st.caption(f"Mode active: `{optimization_mode}`")
        
        st.markdown("**Guardrails**")
        max_cpc_increase_pct = st.slider("Max CPC increase %", 0, 45, 45, 1)

        st.markdown("**Scoring Weights**")
        growth_weight = st.slider("Growth weight", 0.0, 1.0, 0.70, 0.05)
        profit_weight = st.slider("Profitability weight", 0.0, 1.0, 0.30, 0.05)

        min_bids_channel_state = st.slider("Min bids for reliable channel-state", 1, 20, 5, 1)
        cpc_penalty_weight = st.slider("CPC penalty", 0.0, 1.5, 0.65, 0.05)
        min_intent_for_scale = st.slider("Min intent to allow positive scaling", 0.0, 1.0, 0.65, 0.01)
        roe_pullback_floor = st.slider("ROE severe pullback floor", -1.0, 0.5, -0.45, 0.01)
        cr_pullback_ceiling = st.slider("Combined ratio severe pullback ceiling", 0.8, 1.5, 1.35, 0.01)
        max_perf_drop = st.slider("Max performance drop vs current", 0.00, 0.60, 0.15, 0.01)
        min_new_performance = st.slider("Minimum new performance", 0.20, 1.50, 0.80, 0.01)

        st.markdown("**Stat Sig Rules**")
        min_clicks_intent_sig = st.slider("Min clicks for intent significance", 10, 300, 80, 5)
        min_bids_price_sig = st.slider("Min bids for price-test significance", 10, 500, 75, 5)
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
            strategy_path = DEFAULT_PATHS["state_strategy"]
            state_path = DEFAULT_PATHS["state_data"]
            state_seg_path = DEFAULT_PATHS["state_seg"]
            channel_group_path = DEFAULT_PATHS["channel_group"]
            price_path = DEFAULT_PATHS["channel_price_exp"]
            channel_state_path = DEFAULT_PATHS["channel_state"]
            mtimes = (
                file_mtime(strategy_path),
                file_mtime(state_path),
                file_mtime(state_seg_path),
                file_mtime(channel_group_path),
                file_mtime(price_path),
                file_mtime(channel_state_path),
            )
            rec_df, state_df, state_seg_df, price_eval, state_extra_df, state_seg_extra_df = build_all_from_paths(
                strategy_path,
                state_path,
                state_seg_path,
                channel_group_path,
                price_path,
                channel_state_path,
                settings,
                mtimes,
            )
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
            state_df = prepare_state(state_raw, strategy_df, settings)
            state_seg_df = prepare_state_seg(state_seg_raw, state_raw, settings)
            channel_state_df = prepare_channel_state(channel_state_raw)
            price_eval, best_adj = prepare_price_exploration(price_raw, settings)
            rec_df, state_extra_df, state_seg_extra_df, _ = build_model_tables(
                state_df, state_seg_df, channel_state_df, best_adj, price_eval, settings
            )
        else:
            mtimes = (
                file_mtime(strategy_path),
                file_mtime(state_path),
                file_mtime(state_seg_path),
                file_mtime(channel_group_path),
                file_mtime(price_path),
                file_mtime(channel_state_path),
            )
            rec_df, state_df, state_seg_df, price_eval, state_extra_df, state_seg_extra_df = build_all_from_paths(
                strategy_path,
                state_path,
                state_seg_path,
                channel_group_path,
                price_path,
                channel_state_path,
                settings,
                mtimes,
            )
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    rec_df_model = rec_df.copy()
    if "bid_overrides" not in st.session_state:
        st.session_state["bid_overrides"] = load_overrides_from_disk()
    rec_df = apply_user_bid_overrides(rec_df, price_eval, settings, st.session_state["bid_overrides"])
    state_extra_df, state_seg_extra_df, channel_summary_df = summarize_from_rec(rec_df)

    tab_labels = [
        "🏁 Tab 0: Executive State View",
        "🗺️ Tab 1: State Momentum Map",
        "📊 Tab 2: Channel Group Analysis",
        "🧠 Tab 3: Channel Group and States",
        "🧪 Tab 4: Price Exploration Details",
        "📚 Tab 5: General Analytics",
        "🌌 Neon Insights Cockpit",
    ]
    selected_tab = st.radio(
        "View",
        options=tab_labels,
        horizontal=True,
        key="main_view_tab",
    )

    if selected_tab == tab_labels[0]:
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

        render_kpi_tiles(
            [
                {"label": "Total Clicks", "value": f"{total_clicks:,.0f}"},
                {"label": "Avg Win Rate", "value": "n/a" if pd.isna(avg_win_rate) else f"{avg_win_rate:.1%}"},
                {"label": "Cost", "value": f"${total_cost:,.0f}"},
                {"label": "Avg Q2B", "value": "n/a" if pd.isna(avg_q2b) else f"{avg_q2b:.1%}"},
                {"label": "Binds", "value": f"{total_binds:,.0f}"},
                {"label": "CPB", "value": "n/a" if pd.isna(cpb) else f"${cpb:,.0f}"},
                {"label": "ROE", "value": "n/a" if pd.isna(roe_w) else f"{roe_w:.1%}"},
                {"label": "Combined Ratio", "value": "n/a" if pd.isna(cr_w) else f"{cr_w:.1%}"},
                {"label": "LTV", "value": "n/a" if pd.isna(ltv_w) else f"${ltv_w:,.0f}"},
                {"label": "Additional Clicks", "value": f"{add_clicks:,.0f}"},
                {"label": "Additional Binds", "value": f"{add_binds:,.1f}"},
                {"label": "Required Budget", "value": f"${add_budget:,.0f}"},
            ],
            cols=4,
        )

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

        st.markdown("**State Strategy vs Actual Indicator**")
        indicator_view = map_df0[[
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
            map_df0.groupby(["ROE Performance Group", "Performance Stat Sig"], as_index=False)
            .agg(
                States=("State", lambda x: ", ".join(sorted(set(x)))),
                Rows=("State", "count"),
                Binds=("Binds", "sum"),
            )
            .sort_values(["Performance Stat Sig", "Rows"], ascending=[False, False])
        )
        render_formatted_table(state_perf_layer, use_container_width=True)

    elif selected_tab == tab_labels[1]:
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
                render_kpi_tiles(
                    [
                        {"label": "ROE", "value": f"{row['ROE'].iloc[0]:.1%}"},
                        {"label": "Combined Ratio", "value": f"{row['Combined Ratio'].iloc[0]:.1%}"},
                        {"label": "Binds", "value": f"{row['Binds'].iloc[0]:,.0f}"},
                        {"label": "Avg LTV", "value": f"${row['Avg. MRLTV'].iloc[0]:,.0f}"},
                    ],
                    cols=4,
                )
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
                render_kpi_tiles(
                    [
                        {"label": "Current Clicks", "value": f"{state_current_clicks:,.0f}"},
                        {
                            "label": "Additional Clicks",
                            "value": f"{state_add_clicks:,.0f}",
                            "sub": "n/a" if pd.isna(state_add_clicks_pct) else f"{state_add_clicks_pct:.1%}",
                        },
                        {"label": "Additional Binds", "value": f"{state_add_binds:,.1f}"},
                        {"label": "Additional Budget Needed", "value": f"${state_add_budget:,.0f}"},
                        {"label": "Budget Impact", "value": "n/a" if pd.isna(state_add_budget_pct) else f"{state_add_budget_pct:.1%}"},
                    ],
                    cols=5,
                )

                st.markdown("**🧩 Per-Segment KPI + Opportunity**")
                seg_show = seg_view[[
                    "Segment", "ROE Performance Group",
                    "Bids", "Avg. CPC", "Win Rate", "Q2B", "Clicks", "Binds", "Clicks to Binds", "ROE", "Combined Ratio", "Avg. MRLTV",
                    "Expected_Additional_Clicks", "Expected_Additional_Binds", "Additional Budget Required"
                ]].sort_values("Expected_Additional_Clicks", ascending=False)
                render_formatted_table(seg_show, use_container_width=True)

                st.caption(
                    f"Recommendation Strategy: `{st.session_state.get('global_optimization_mode', settings.optimization_mode)}` | "
                    f"CPC cap: {effective_cpc_cap_pct(settings):.0f}%"
                )

                st.markdown("**📌 Channel Groups In This State**")
                state_channels = rec_df[rec_df["State"] == selected_state].copy()
                state_channels_model = rec_df_model[rec_df_model["State"] == selected_state].copy()
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
                            **{"Expected Additional Clicks": ("Expected Additional Clicks", "sum")},
                            **{"Expected Additional Binds": ("Expected Additional Binds", "sum")},
                            **{"CPC Lift %": ("CPC Lift %", "mean")},
                        ).sort_values("Expected Additional Clicks", ascending=False)
                        rec_bid_model = state_channels_model.groupby("Channel Groups", as_index=False).agg(
                            **{"Rec. Bid Adj.": ("Applied Price Adjustment %", "median")}
                        )
                        cg_state = cg_state.merge(rec_bid_model, on="Channel Groups", how="left")
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
                        popup_state_df = precompute_popup_options_for_state(rec_df, price_eval, selected_state, settings)
                        table_df["Selected Price Adj."] = table_df["Rec. Bid Adj."]
                        table_df["Selection Source"] = "Suggested"
                        table_df["Adj Selection"] = ""
                        table_df["Adj Options"] = [[] for _ in range(len(table_df))]
                        table_df["Adj Options JSON"] = ["[]" for _ in range(len(table_df))]
                        table_df["Adj Map JSON"] = ["{}" for _ in range(len(table_df))]
                        for idx, rr in table_df.iterrows():
                            okey = f"{selected_state}|{rr['Channel Groups']}"
                            ov = st.session_state["bid_overrides"].get(okey, {})
                            if isinstance(ov, dict) and ov.get("apply", False):
                                table_df.at[idx, "Selected Price Adj."] = float(ov.get("requested_adj", ov.get("adj", rr["Rec. Bid Adj."])))
                                table_df.at[idx, "Selection Source"] = "Manual adjustment"
                            ch = str(rr["Channel Groups"])
                            ch_opts = popup_state_df[popup_state_df["Channel Groups"] == ch] if not popup_state_df.empty else pd.DataFrame()
                            labels = []
                            label_to_adj = {}
                            for _, op in ch_opts.iterrows():
                                ladj = float(op.get("Bid Adj %", 0) or 0)
                                ll = format_adj_option_label(
                                    ladj,
                                    float(op.get("Win Rate Uplift", 0) or 0),
                                    float(op.get("CPC Uplift", 0) or 0),
                                    op.get("CPB Impact", np.nan),
                                    str(op.get("Sig Level", "")),
                                )
                                labels.append(ll)
                                label_to_adj[ll] = ladj
                            if not labels:
                                base_adj = float(table_df.at[idx, "Selected Price Adj."])
                                ll = f"{base_adj:+.0f}%: n/a Clicks || n/a CPC || n/a CPB (no stat-sig)"
                                labels = [ll]
                                label_to_adj[ll] = base_adj
                            current_adj = float(table_df.at[idx, "Selected Price Adj."])
                            has_current = any(
                                close_adj(as_float(label_to_adj.get(lb, parse_adj_from_label(lb))), current_adj)
                                for lb in labels
                            )
                            if not has_current:
                                fallback = f"{current_adj:+.0f}%: model recommendation (no matched stat-sig option)"
                                labels = [fallback] + labels
                                label_to_adj[fallback] = current_adj
                            table_df.at[idx, "Adj Options"] = labels
                            table_df.at[idx, "Adj Options JSON"] = json.dumps(labels)
                            table_df.at[idx, "Adj Map JSON"] = json.dumps(label_to_adj)
                            selected_label = next(
                                (
                                    lb
                                    for lb in labels
                                    if close_adj(as_float(label_to_adj.get(lb, parse_adj_from_label(lb))), current_adj)
                                ),
                                labels[0],
                            )
                            table_df.at[idx, "Adj Selection"] = selected_label
                            # For manual selections, force row-level expected metrics from the same
                            # state+channel adjustment candidate set used in the dropdown.
                            if table_df.at[idx, "Selection Source"] == "Manual adjustment" and not ch_opts.empty:
                                m = ch_opts.copy()
                                m["dist"] = (pd.to_numeric(m["Bid Adj %"], errors="coerce").fillna(0.0) - current_adj).abs()
                                best = m.sort_values(["dist", "Bid Adj %"], ascending=[True, True]).iloc[0]
                                table_df.at[idx, "Expected Additional Clicks"] = float(best.get("Additional Clicks", table_df.at[idx, "Expected Additional Clicks"]) or 0.0)
                                table_df.at[idx, "Expected Additional Binds"] = float(best.get("Additional Binds", table_df.at[idx, "Expected Additional Binds"]) or 0.0)
                                table_df.at[idx, "CPC Lift %"] = float(best.get("CPC Uplift", table_df.at[idx, "CPC Lift %"]) or 0.0)
                                table_df.at[idx, "Expected Total Cost"] = float(best.get("Expected Total Cost", table_df.at[idx, "Expected Total Cost"]) or 0.0)
                                table_df.at[idx, "Additional Budget Needed"] = float(best.get("Additional Budget Needed", table_df.at[idx, "Additional Budget Needed"]) or 0.0)
                        table_df = table_df[
                            [
                                "Channel Groups",
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
                                "Selection Source",
                                "Adj Options",
                                "Adj Options JSON",
                                "Adj Map JSON",
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

                        edited = grid_df.copy()
                        draft_key = f"tab1_grid_draft_{selected_state}"
                        prev = st.session_state.get(draft_key)
                        if isinstance(prev, pd.DataFrame) and "Channel Groups" in prev.columns:
                            if set(prev["Channel Groups"].astype(str)) == set(edited["Channel Groups"].astype(str)):
                                edited = prev.copy()
                        # Use AG Grid for true row-level dropdown options in-table.
                        use_aggrid_for_state_table = AGGRID_AVAILABLE
                        if AGGRID_AVAILABLE and use_aggrid_for_state_table:
                            gb = GridOptionsBuilder.from_dataframe(edited)
                            gb.configure_default_column(resizable=True, sortable=True, filter=True)
                            gb.configure_grid_options(singleClickEdit=True, stopEditingWhenCellsLoseFocus=True)
                            gb.configure_column("Channel Groups", editable=False, pinned="left", width=180)
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
                                cellEditorPopup=True,
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
                            gb.configure_column("Selection Source", editable=False, width=108)
                            gb.configure_column("Adj Options", hide=True)
                            gb.configure_column("Adj Options JSON", hide=True)
                            gb.configure_column("Adj Map JSON", hide=True)
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
                                height=620,
                                theme="balham-dark",
                                custom_css=custom_css,
                                key=f"tab1_aggrid_{selected_state}_{st.session_state.get(f'tab1_grid_refresh_{selected_state}', 0)}",
                            )
                            edited = pd.DataFrame(grid["data"])
                            st.session_state[draft_key] = edited
                        else:
                            edited = st.data_editor(
                                edited,
                                use_container_width=True,
                                hide_index=True,
                                key=f"tab1_apply_editor_{selected_state}",
                                column_config={
                                    "Channel Groups": st.column_config.TextColumn("Channel Groups", disabled=True),
                                    "Bids": st.column_config.NumberColumn("Bids", format="localized", disabled=True),
                                    "SOV": st.column_config.NumberColumn("SOV", format="%.0f%%", disabled=True),
                                    "Clicks": st.column_config.NumberColumn("Clicks", format="localized", disabled=True),
                                    "Binds": st.column_config.NumberColumn("Binds", format="%.2f", disabled=True),
                                    "Win Rate": st.column_config.NumberColumn("Win Rate", format="%.2f%%", disabled=True),
                                    "Total Cost": st.column_config.NumberColumn("Total Cost", format="dollar", disabled=True),
                                    "Rec Bid Adj": st.column_config.NumberColumn("Rec. Bid Adj.", format="%+.0f%%", disabled=True),
                                    "Adj Selection": st.column_config.TextColumn("Adj Selection", disabled=True),
                                    "Selected Price Adj": st.column_config.NumberColumn("Selected Price Adj.", format="%+.0f%%", disabled=True),
                                    "Expected Total Cost": st.column_config.NumberColumn("Expected Total Cost", format="dollar", disabled=True),
                                    "Additional Budget Needed": st.column_config.NumberColumn("Adjusted Budget", format="dollar", disabled=True),
                                    "Expected Additional Clicks": st.column_config.NumberColumn("Expected Additional Clicks", format="localized", disabled=True),
                                    "Expected Additional Binds": st.column_config.NumberColumn("Expected Additional Binds", format="%.2f", disabled=True),
                                    "CPC Lift %": st.column_config.NumberColumn("CPC Lift %", format="%.0f%%", disabled=True),
                                    "Selection Source": st.column_config.TextColumn("Selection Source", disabled=True),
                                    "Adj Options": None,
                                    "Adj Options JSON": None,
                                    "Adj Map JSON": None,
                                },
                            )
                            st.session_state[draft_key] = edited
                        # Stage dropdown selections in rows (apply on Save).
                        for i, rr in edited.iterrows():
                            adj_map = {}
                            try:
                                adj_map = json.loads(rr.get("Adj Map JSON", "{}") or "{}")
                            except Exception:
                                adj_map = {}
                            sel_label = rr.get("Adj Selection", "")
                            adj = as_float(adj_map.get(sel_label, parse_adj_from_label(sel_label)), np.nan)
                            if pd.notna(adj):
                                rec_adj = as_float(rr.get("Rec Bid Adj", 0.0), 0.0)
                                edited.at[i, "Selected Price Adj"] = float(adj)
                                if not close_adj(float(adj), rec_adj):
                                    edited.at[i, "Selection Source"] = "Manual adjustment"
                                else:
                                    edited.at[i, "Selection Source"] = "Suggested"

                                # Keep expected columns in sync with selected adjustment for preview.
                                ch = str(rr.get("Channel Groups", ""))
                                ch_opts = popup_state_df[popup_state_df["Channel Groups"] == ch] if not popup_state_df.empty else pd.DataFrame()
                                if not ch_opts.empty:
                                    mm = ch_opts.copy()
                                    mm["dist"] = (pd.to_numeric(mm["Bid Adj %"], errors="coerce").fillna(0.0) - float(adj)).abs()
                                    best = mm.sort_values(["dist", "Bid Adj %"], ascending=[True, True]).iloc[0]
                                    edited.at[i, "Expected Additional Clicks"] = as_float(best.get("Additional Clicks"), as_float(rr.get("Expected Additional Clicks"), 0.0))
                                    edited.at[i, "Expected Additional Binds"] = as_float(best.get("Additional Binds"), as_float(rr.get("Expected Additional Binds"), 0.0))
                                    edited.at[i, "CPC Lift %"] = as_float(best.get("CPC Uplift"), as_float(rr.get("CPC Lift %"), 0.0))
                                    edited.at[i, "Expected Total Cost"] = as_float(best.get("Expected Total Cost"), as_float(rr.get("Expected Total Cost"), 0.0))
                                    edited.at[i, "Additional Budget Needed"] = as_float(best.get("Additional Budget Needed"), as_float(rr.get("Additional Budget Needed"), 0.0))
                        st.session_state[draft_key] = edited
                        saving_key = f"tab1_is_saving_{selected_state}"
                        is_saving = bool(st.session_state.get(saving_key, False))
                        if is_saving:
                            st.info("Saving changes and recalculating. Please wait...")

                        a_save, a_reset = st.columns([1, 1])
                        do_save_click = a_save.button("Save Edits", key=f"tab1_save_edits_{selected_state}", disabled=is_saving)
                        do_reset_click = a_reset.button(
                            "Reset To Recommended Bid",
                            key=f"tab1_reset_rec_{selected_state}",
                            disabled=is_saving,
                        )

                        if do_save_click:
                            st.session_state["tab1_modal_pending_action"] = "save"
                            st.session_state["tab1_modal_confirmed"] = False
                            st.rerun()
                        if do_reset_click:
                            st.session_state["tab1_modal_pending_action"] = "reset"
                            st.session_state["tab1_modal_confirmed"] = False
                            st.rerun()

                        pending_action = str(st.session_state.get("tab1_modal_pending_action", ""))
                        if pending_action == "save":
                            confirm_action_modal(
                                "Confirm Save",
                                "Apply and save all changes in this state table?",
                                f"tab1_confirm_save_{selected_state}",
                                f"tab1_cancel_save_{selected_state}",
                            )
                        elif pending_action == "reset":
                            confirm_action_modal(
                                "Confirm Reset",
                                "Reset all rows in this state table to recommended bids and save?",
                                f"tab1_confirm_reset_{selected_state}",
                                f"tab1_cancel_reset_{selected_state}",
                            )

                        run_action = bool(st.session_state.get("tab1_modal_confirmed", False))
                        if run_action and pending_action == "reset":
                            st.session_state[saving_key] = True
                            with st.status("Processing reset...", expanded=True) as status:
                                status.write("Saving data...")
                                edited["Selection Source"] = "Suggested"
                                edited["Selected Price Adj"] = pd.to_numeric(edited["Rec Bid Adj"], errors="coerce").fillna(0.0)
                                if "Adj Selection" in edited.columns and "Adj Options" in edited.columns:
                                    for idx in edited.index:
                                        opts = edited.at[idx, "Adj Options"]
                                        recv = float(pd.to_numeric(pd.Series([edited.at[idx, "Rec Bid Adj"]]), errors="coerce").fillna(0.0).iloc[0])
                                        if isinstance(opts, list) and opts:
                                            picked = next((lb for lb in opts if parse_adj_from_label(lb) == recv), opts[0])
                                            edited.at[idx, "Adj Selection"] = picked
                                prev_overrides = dict(st.session_state.get("bid_overrides", {}))
                                new_overrides = {
                                    k: v for k, v in prev_overrides.items() if not str(k).startswith(f"{selected_state}|")
                                }
                                ok, err = save_overrides_to_disk(new_overrides)
                                status.write("Recalculating binds prediction...")
                                time.sleep(0.15)
                                status.write("Collecting data...")
                                time.sleep(0.15)
                                status.update(label="Reset completed.", state="complete")

                            st.session_state[saving_key] = False
                            if not ok:
                                st.error(err)
                            st.session_state["bid_overrides"] = new_overrides
                            st.session_state[draft_key] = edited
                            st.session_state["tab1_save_notice"] = "Reset to recommended bids completed."
                            st.session_state.pop(f"tab1_grid_draft_{selected_state}", None)
                            st.session_state[f"tab1_grid_refresh_{selected_state}"] = int(
                                st.session_state.get(f"tab1_grid_refresh_{selected_state}", 0)
                            ) + 1
                            st.session_state["tab1_modal_pending_action"] = ""
                            st.session_state["tab1_modal_confirmed"] = False
                            st.rerun()

                        if run_action and pending_action == "save":
                            st.session_state[saving_key] = True
                            prev_overrides = dict(st.session_state.get("bid_overrides", {}))
                            new_overrides = dict(prev_overrides)
                            audit_rows = []
                            for _, rr in edited.iterrows():
                                okey = f"{selected_state}|{rr['Channel Groups']}"
                                adj_map = {}
                                try:
                                    adj_map = json.loads(rr.get("Adj Map JSON", "{}") or "{}")
                                except Exception:
                                    adj_map = {}
                                sel_label = rr.get("Adj Selection", "")
                                adj_from_dropdown = adj_map.get(sel_label, parse_adj_from_label(sel_label))
                                rec_adj = as_float(rr.get("Rec Bid Adj", 0.0), 0.0)
                                prev = prev_overrides.get(okey, {})
                                prev_adj = as_float(prev.get("adj", rec_adj), rec_adj) if isinstance(prev, dict) else rec_adj
                                selected_adj = as_float(rr.get("Selected Price Adj", rec_adj), rec_adj)
                                req_adj = as_float(adj_from_dropdown, selected_adj)
                                is_manual = (not close_adj(req_adj, rec_adj))
                                if is_manual:
                                    applied_adj, msg = nearest_available_adj(rr["Channel Groups"], req_adj, popup_state_df)
                                    new_overrides[okey] = {
                                        "apply": True,
                                        "adj": float(applied_adj),
                                        "requested_adj": float(req_adj),
                                        "source": "manual",
                                    }
                                    audit_rows.append(
                                        {
                                            "State": selected_state,
                                            "Channel Groups": rr["Channel Groups"],
                                            "Previous Adj %": prev_adj,
                                            "Requested Adj %": req_adj,
                                            "Applied Adj %": applied_adj,
                                            "Status": "Saved",
                                            "Note": msg,
                                        }
                                    )
                                else:
                                    new_overrides.pop(okey, None)
                            changed_keys = set(prev_overrides.keys()) ^ set(new_overrides.keys())
                            for k in set(prev_overrides.keys()) & set(new_overrides.keys()):
                                a = prev_overrides.get(k, {})
                                b = new_overrides.get(k, {})
                                if (
                                    bool(a.get("apply", False)) != bool(b.get("apply", False))
                                    or (not close_adj(as_float(a.get("adj", 0.0), 0.0), as_float(b.get("adj", 0.0), 0.0)))
                                    or (not close_adj(as_float(a.get("requested_adj", a.get("adj", 0.0)), 0.0), as_float(b.get("requested_adj", b.get("adj", 0.0)), 0.0)))
                                ):
                                    changed_keys.add(k)
                            changed_rows = len(changed_keys)
                            with st.status("Processing save...", expanded=True) as status:
                                status.write("Saving data...")
                                st.session_state["bid_overrides"] = new_overrides
                                ok, err = save_overrides_to_disk(new_overrides)
                                status.write("Recalculating binds prediction...")
                                time.sleep(0.15)
                                status.write("Collecting data...")
                                time.sleep(0.15)
                                status.update(label="Save completed.", state="complete")
                            st.session_state[saving_key] = False
                            if not ok:
                                st.error(err)
                            st.session_state["tab1_save_notice"] = f"Saved {changed_rows} manual adjustments."
                            if audit_rows:
                                st.session_state["tab1_save_audit"] = pd.DataFrame(audit_rows)
                            st.session_state.pop(f"tab1_grid_draft_{selected_state}", None)
                            st.session_state[f"tab1_grid_refresh_{selected_state}"] = int(
                                st.session_state.get(f"tab1_grid_refresh_{selected_state}", 0)
                            ) + 1
                            st.session_state["tab1_modal_pending_action"] = ""
                            st.session_state["tab1_modal_confirmed"] = False
                            st.rerun()

                        if st.session_state.get("tab1_save_notice"):
                            st.success(st.session_state.pop("tab1_save_notice"))
                        audit_df = st.session_state.get("tab1_save_audit")
                        if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
                            st.markdown("**Save Audit (latest action)**")
                            render_formatted_table(audit_df, use_container_width=True)
                        st.caption("Use `Adj Selection` dropdown in the table, then click `Save Edits` to apply all changes.")

    elif selected_tab == tab_labels[2]:
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

    elif selected_tab == tab_labels[3]:
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

    elif selected_tab == tab_labels[4]:
        st.subheader("🧪 Price Exploration Details")
        st.caption("Master-detail view of valid test points by state + channel group.")
        st.markdown(
            """
            <style>
            .px-card {
                background:
                    radial-gradient(120% 140% at 0% 0%, rgba(14,165,233,0.16), rgba(14,165,233,0) 42%),
                    radial-gradient(120% 120% at 100% 0%, rgba(34,211,238,0.12), rgba(34,211,238,0) 44%),
                    linear-gradient(145deg, rgba(15,23,42,0.95), rgba(17,24,39,0.90));
                border: 1px solid rgba(56,189,248,0.22);
                border-radius: 14px;
                padding: 12px 12px 10px 12px;
                margin-bottom: 8px;
                box-shadow: 0 0 0 1px rgba(15,23,42,0.2), 0 12px 30px rgba(2,6,23,0.42);
                transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
            }
            .px-card:hover {
                transform: translateY(-1px);
                border-color: rgba(34,211,238,0.45);
                box-shadow: 0 0 0 1px rgba(15,23,42,0.16), 0 14px 34px rgba(2,6,23,0.5);
            }
            .px-card-selected {
                border-color: rgba(52,211,153,0.9);
                box-shadow:
                    0 0 0 1px rgba(16,185,129,0.55),
                    0 0 22px rgba(16,185,129,0.3),
                    0 10px 28px rgba(2,6,23,0.5);
            }
            .px-title {
                font-size: 0.95rem;
                font-weight: 700;
                color: #e2e8f0;
                margin-bottom: 4px;
            }
            .px-sub {
                font-size: 0.79rem;
                color: #bfdbfe;
            }
            .px-points {
                font-size: 0.78rem;
                color: #a7f3d0;
                margin-top: 7px;
                word-break: break-word;
            }
            .px-chip-row {
                margin-top: 8px;
                display: flex;
                gap: 6px;
                flex-wrap: wrap;
            }
            .px-chip {
                display: inline-block;
                border: 1px solid rgba(148,163,184,0.35);
                border-radius: 999px;
                padding: 2px 8px;
                font-size: 0.72rem;
                color: #cbd5e1;
                background: rgba(15,23,42,0.58);
            }
            .px-detail-shell {
                border: 1px solid rgba(56,189,248,0.22);
                border-radius: 14px;
                padding: 12px 14px;
                margin-bottom: 10px;
                background:
                    radial-gradient(120% 120% at 0% 0%, rgba(14,165,233,0.12), rgba(14,165,233,0) 48%),
                    linear-gradient(145deg, rgba(15,23,42,0.90), rgba(17,24,39,0.85));
            }
            [class*="st-key-px_card_"] button {
                background: linear-gradient(145deg, rgba(19,27,44,0.96), rgba(21,30,48,0.90)) !important;
                border: 1px solid rgba(45,212,191,0.78) !important;
                border-radius: 14px !important;
                padding: 7px 9px !important;
                min-height: 88px !important;
                height: auto !important;
                text-align: left !important;
                color: #e2e8f0 !important;
                font-size: 0.79rem !important;
                line-height: 1.28 !important;
                box-shadow: 0 0 0 1px rgba(45,212,191,0.22), 0 0 18px rgba(45,212,191,0.28), 0 10px 26px rgba(2,6,23,0.35) !important;
                margin-bottom: 6px !important;
                white-space: pre-line !important;
            }
            [class*="st-key-px_card_"] button > div {
                width: 100% !important;
                display: flex !important;
                justify-content: flex-start !important;
                align-items: flex-start !important;
                text-align: left !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            [class*="st-key-px_card_"] [data-testid="stButton"] {
                margin: 0 !important;
                padding: 0 !important;
                text-indent: 0 !important;
            }
            [class*="st-key-px_card_"] button p {
                margin: 0 !important;
                line-height: 1.30 !important;
                letter-spacing: 0.1px;
                text-indent: 0 !important;
                padding-left: 0 !important;
                width: 100% !important;
                text-align: left !important;
            }
            [class*="st-key-px_card_"] button strong {
                color: #f8fafc !important;
                font-weight: 800 !important;
                font-size: 0.98rem !important;
            }
            [class*="st-key-px_card_"] button em {
                color: #67e8f9 !important;
                font-style: normal !important;
                font-weight: 700 !important;
            }
            [class*="st-key-tab4_cards_scroll"] [data-testid="stVerticalBlock"]::-webkit-scrollbar {
                width: 10px;
            }
            [class*="st-key-tab4_cards_scroll"] {
                border: 1px solid rgba(45,212,191,0.62) !important;
                border-radius: 14px !important;
                box-shadow: 0 0 0 1px rgba(45,212,191,0.18), 0 8px 24px rgba(2,6,23,0.35);
                background: linear-gradient(145deg, rgba(10,16,28,0.64), rgba(10,16,28,0.38));
            }
            [class*="st-key-tab4_right_scroll"] {
                border: 1px solid rgba(45,212,191,0.62) !important;
                border-radius: 14px !important;
                box-shadow: 0 0 0 1px rgba(45,212,191,0.18), 0 8px 24px rgba(2,6,23,0.35);
                background: linear-gradient(145deg, rgba(10,16,28,0.64), rgba(10,16,28,0.38));
            }
            [class*="st-key-tab4_detail_shell_"] {
                border: 1px solid rgba(56,189,248,0.22) !important;
                border-radius: 14px !important;
                box-shadow: 0 0 0 1px rgba(45,212,191,0.10), 0 8px 22px rgba(2,6,23,0.22);
                background:
                    radial-gradient(120% 120% at 0% 0%, rgba(14,165,233,0.12), rgba(14,165,233,0) 48%),
                    linear-gradient(145deg, rgba(15,23,42,0.90), rgba(17,24,39,0.85));
            }
            [class*="st-key-tab4_cards_scroll"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-track {
                background: rgba(15,23,42,0.55);
                border-radius: 999px;
            }
            [class*="st-key-tab4_cards_scroll"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, rgba(45,212,191,0.85), rgba(56,189,248,0.85));
                border-radius: 999px;
                border: 2px solid rgba(15,23,42,0.65);
            }
            [class*="st-key-tab4_right_scroll"] [data-testid="stVerticalBlock"]::-webkit-scrollbar {
                width: 10px;
            }
            [class*="st-key-tab4_right_scroll"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-track {
                background: rgba(15,23,42,0.55);
                border-radius: 999px;
            }
            [class*="st-key-tab4_right_scroll"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, rgba(45,212,191,0.85), rgba(56,189,248,0.85));
                border-radius: 999px;
                border: 2px solid rgba(15,23,42,0.65);
            }
            [class*="st-key-px_card_"] button:hover {
                border-color: rgba(45,212,191,1.0) !important;
                transform: translateY(-1px);
            }
            [class*="st-key-px_card_"] button[kind="primary"] {
                border-color: rgba(16,185,129,0.95) !important;
                box-shadow: 0 0 0 1px rgba(16,185,129,0.52), 0 0 18px rgba(16,185,129,0.27), 0 10px 28px rgba(2,6,23,0.5) !important;
            }
            .px-sep {
                height: 1px;
                background: linear-gradient(90deg, rgba(148,163,184,0.06), rgba(45,212,191,0.7), rgba(148,163,184,0.06));
                margin: 12px 0 12px 0;
            }
            .px-subhead {
                color: #c7d2fe;
                font-size: 0.90rem;
                font-weight: 700;
                margin: 2px 0 8px 0;
                letter-spacing: 0.2px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        master_df, detail_df = build_price_exploration_master_detail(rec_df, price_eval, settings)
        if master_df.empty or detail_df.empty:
            st.info("No stat-significant price exploration points found for the current data.")
        else:
            fx1, fx2, fx3 = st.columns(3)
            state_opts = sorted(master_df["State"].dropna().unique().tolist())
            channel_opts = sorted(master_df["Channel Groups"].dropna().unique().tolist())
            segment_opts = sorted(master_df["Segment"].dropna().unique().tolist())
            sel_state_px = fx1.multiselect("State", options=state_opts, default=state_opts, key="tab4_px_state")
            sel_channel_px = fx2.multiselect("Channel Group", options=channel_opts, default=channel_opts, key="tab4_px_channel")
            sel_segment_px = fx3.multiselect("Segment", options=segment_opts, default=segment_opts, key="tab4_px_segment")

            filt_master = master_df[
                master_df["State"].isin(sel_state_px)
                & master_df["Channel Groups"].isin(sel_channel_px)
                & master_df["Segment"].isin(sel_segment_px)
            ].copy()
            if filt_master.empty:
                st.warning("No cards match the selected filters.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Cards", f"{len(filt_master):,.0f}")
                m2.metric("Total Binds", f"{filt_master['Total Binds'].sum():,.0f}")
                m3.metric("Total Clicks", f"{filt_master['Total Clicks'].sum():,.0f}")

                filt_master = filt_master.sort_values(["Total Binds", "Total Bids"], ascending=False)
                quick_find = st.text_input(
                    "Search cards (state or channel group)",
                    value="",
                    key="tab4_px_search",
                    placeholder="Type state code or channel group...",
                ).strip()
                if quick_find:
                    q = quick_find.lower()
                    filt_master = filt_master[
                        filt_master["State"].astype(str).str.lower().str.contains(q, regex=False)
                        | filt_master["Channel Groups"].astype(str).str.lower().str.contains(q, regex=False)
                    ].copy()
                    if filt_master.empty:
                        st.warning("No cards after search filter.")
                        st.stop()

                valid_keys = set(
                    filt_master.apply(lambda r: f"{r['State']}|{r['Channel Groups']}|{r['Segment']}", axis=1).tolist()
                )
                selected_key = st.session_state.get("px_selected_card_key")
                if selected_key not in valid_keys:
                    first_row = filt_master.iloc[0]
                    selected_key = f"{first_row['State']}|{first_row['Channel Groups']}|{first_row['Segment']}"
                    st.session_state["px_selected_card_key"] = selected_key

                p1, p2, p3 = st.columns([1, 1, 2])
                page_size = p1.selectbox("Cards per page", options=[10, 25, 40, 60], index=1, key="tab4_px_page_size")
                total_cards = int(len(filt_master))
                page_count = max((total_cards + int(page_size) - 1) // int(page_size), 1)
                page_num = int(p2.number_input("Page", min_value=1, max_value=page_count, value=1, step=1, key="tab4_px_page_num"))
                start_idx = (page_num - 1) * int(page_size)
                end_idx = min(start_idx + int(page_size), total_cards)
                p3.caption(f"Showing cards {start_idx + 1:,} - {end_idx:,} of {total_cards:,}")
                page_df = filt_master.iloc[start_idx:end_idx].copy()
                # Panel height follows left card volume; both sections use the same height.
                left_target = int(102 * len(page_df) + 100)
                panel_height = max(980, min(left_target, 1900))
                st.markdown("**Price Exploration Alert**")
                left, right = st.columns([1.1, 1.9], gap="large")
                with left:
                    with fixed_height_container(panel_height, key="tab4_cards_scroll"):
                        for _, r in page_df.iterrows():
                            ch_raw = str(r["Channel Groups"])
                            ch_name = " ".join(ch_raw.split())
                            card_key = f"{r['State']}|{ch_raw}|{r['Segment']}"
                            active = card_key == st.session_state.get("px_selected_card_key")
                            points = [p.strip() for p in str(r["Testing Points"]).split("||") if str(p).strip()]
                            point_boxes = " ".join([f"[{p}]" for p in points[:7]])
                            header_line = f"**{ch_name} · {r['State']}**"
                            stats_line = (
                                f"**Bids:** {r['Total Bids']:,.0f}   |   "
                                f"**Clicks:** {r['Total Clicks']:,.0f}   |   "
                                f"**Binds:** {r['Total Binds']:,.0f}"
                            )
                            card_label = (
                                f"{header_line}\n"
                                f"{stats_line}\n"
                                f"*Testing:* {point_boxes if point_boxes else 'n/a'}"
                            )
                            if st.button(
                                card_label,
                                key=f"px_card_{card_key}",
                                use_container_width=True,
                                type="primary" if active else "secondary",
                            ):
                                st.session_state["px_selected_card_key"] = card_key
                                selected_key = card_key

                # Resolve the selected card after click handling so one click updates details.
                detail_lookup = build_price_exploration_detail_lookup(detail_df)
                state_s, channel_s, segment_s = str(selected_key).split("|", 2)
                sdet_preview = detail_lookup.get(selected_key, pd.DataFrame()).copy()
                sel_rows = rec_df[
                    (rec_df["State"] == state_s)
                    & (rec_df["Channel Groups"] == channel_s)
                    & (rec_df["Segment"] == segment_s)
                ].copy()
                strategy_txt = "n/a"
                if not sel_rows.empty and "Strategy Bucket" in sel_rows.columns:
                    sb = sel_rows["Strategy Bucket"].dropna()
                    strategy_txt = str(sb.iloc[0]) if not sb.empty else "n/a"
                source_txt = "Platform"
                adj_val = np.nan
                okey = f"{state_s}|{channel_s}"
                ov = st.session_state.get("bid_overrides", {}).get(okey, {})
                if isinstance(ov, dict) and bool(ov.get("apply", False)):
                    adj_val = as_float(ov.get("requested_adj", ov.get("adj", np.nan)), np.nan)
                    source_txt = "Manual"
                if pd.isna(adj_val):
                    if not sel_rows.empty and "Suggested Price Adjustment %" in sel_rows.columns:
                        adj_val = float(pd.to_numeric(sel_rows["Suggested Price Adjustment %"], errors="coerce").median())
                    elif not sel_rows.empty and "Applied Price Adjustment %" in sel_rows.columns:
                        adj_val = float(pd.to_numeric(sel_rows["Applied Price Adjustment %"], errors="coerce").median())
                adj_txt = "n/a" if pd.isna(adj_val) else f"{adj_val:+.0f}% ({source_txt})"
                with right:
                    with fixed_height_container(panel_height, key="tab4_right_scroll"):
                        sdet = sdet_preview.copy()
                        if sdet.empty:
                            st.info("No detail points found for the selected card.")
                        else:
                            # Top-right manual adjustment selector (same override model as Tab 1), auto-saved on change.
                            rec_adj = np.nan
                            if not sel_rows.empty and "Suggested Price Adjustment %" in sel_rows.columns:
                                rec_adj = float(pd.to_numeric(sel_rows["Suggested Price Adjustment %"], errors="coerce").median())
                            elif not sel_rows.empty and "Applied Price Adjustment %" in sel_rows.columns:
                                rec_adj = float(pd.to_numeric(sel_rows["Applied Price Adjustment %"], errors="coerce").median())
                            if pd.isna(rec_adj):
                                rec_adj = 0.0

                            override_obj = st.session_state.get("bid_overrides", {}).get(okey, {})
                            manual_req_adj = np.nan
                            if isinstance(override_obj, dict) and bool(override_obj.get("apply", False)):
                                manual_req_adj = as_float(override_obj.get("requested_adj", override_obj.get("adj", np.nan)), np.nan)
                            effective_adj = manual_req_adj if not pd.isna(manual_req_adj) else rec_adj

                            sdet_opts = sdet.sort_values("Bid Adj %").copy()
                            label_to_adj_tab4: dict[str, float] = {}
                            option_labels_tab4: list[str] = []
                            for _, op in sdet_opts.iterrows():
                                adj = as_float(op.get("Bid Adj %", 0.0), 0.0)
                                wr_u = as_float(op.get("Win Rate Uplift", 0.0), 0.0)
                                cpc_u = as_float(op.get("CPC Uplift", 0.0), 0.0)
                                add_b = as_float(op.get("Additional Binds", 0.0), 0.0)
                                sig = str(op.get("Sig Level", "n/a"))
                                lbl = f"{adj:+.0f}%: {wr_u:+.1%} WR || {cpc_u:+.1%} CPC || {add_b:+.2f} Binds ({sig})"
                                option_labels_tab4.append(lbl)
                                label_to_adj_tab4[lbl] = adj

                            # Ensure current effective value is always selectable.
                            if not any(close_adj(v, effective_adj) for v in label_to_adj_tab4.values()):
                                fallback_lbl = f"{effective_adj:+.0f}%: current selection"
                                option_labels_tab4 = [fallback_lbl] + option_labels_tab4
                                label_to_adj_tab4[fallback_lbl] = effective_adj

                            cur_label = next(
                                (lb for lb, v in label_to_adj_tab4.items() if close_adj(v, effective_adj)),
                                option_labels_tab4[0] if option_labels_tab4 else f"{effective_adj:+.0f}%: current selection",
                            )

                            safe_state = re.sub(r"[^A-Za-z0-9]+", "_", str(state_s))
                            safe_channel = re.sub(r"[^A-Za-z0-9]+", "_", str(channel_s))
                            safe_segment = re.sub(r"[^A-Za-z0-9]+", "_", str(segment_s))
                            detail_shell_key = f"tab4_detail_shell_{safe_state}_{safe_channel}_{safe_segment}"
                            with st.container(border=True, key=detail_shell_key):
                                dleft, dright = st.columns([1.45, 1.0])
                                with dleft:
                                    st.markdown(f"**Details: {state_s} · {channel_s} · {segment_s}**")
                                    st.caption(
                                        f"State Strategy: {strategy_txt}  |  Recommended Bid Adjustment: {adj_txt}"
                                    )
                                    st.caption(f"Evidence: {sdet['Evidence Label'].iloc[0]}")
                                with dright:
                                    chosen_label = st.selectbox(
                                        "Bid Adjustment (Manual Override)",
                                        options=option_labels_tab4,
                                        index=max(0, option_labels_tab4.index(cur_label)) if option_labels_tab4 else 0,
                                        key=f"tab4_adj_select_{safe_state}_{safe_channel}",
                                        help="Change bid adjustment for this state + channel. Saved immediately.",
                                    )

                            chosen_adj = as_float(label_to_adj_tab4.get(chosen_label, effective_adj), effective_adj)
                            if not close_adj(chosen_adj, effective_adj):
                                new_overrides = dict(st.session_state.get("bid_overrides", {}))
                                if close_adj(chosen_adj, rec_adj):
                                    new_overrides.pop(okey, None)
                                else:
                                    new_overrides[okey] = {
                                        "apply": True,
                                        "adj": float(chosen_adj),
                                        "requested_adj": float(chosen_adj),
                                        "source": "manual",
                                    }
                                ok_save, err_save = save_overrides_to_disk(new_overrides)
                                if ok_save:
                                    st.session_state["bid_overrides"] = new_overrides
                                    st.session_state["tab4_save_notice"] = "saved"
                                else:
                                    st.session_state["tab4_save_notice"] = err_save or "Failed to save manual override."
                                st.rerun()

                            if st.session_state.get("tab4_save_notice"):
                                msg = st.session_state.pop("tab4_save_notice")
                                if str(msg).lower() == "saved":
                                    st.caption("✓ Saved")
                                else:
                                    st.caption(f"⚠ {msg}")

                            seg_row = state_seg_df[
                                (state_seg_df["State"] == state_s) & (state_seg_df["Segment"] == segment_s)
                            ].head(1)
                            if seg_row.empty:
                                seg_roe = float(pd.to_numeric(sdet.get("ROE Proxy", np.nan), errors="coerce").mean())
                                seg_ltv = float(pd.to_numeric(sdet.get("MRLTV Proxy", np.nan), errors="coerce").mean())
                                seg_cpb = np.nan
                                seg_binds = float(pd.to_numeric(sdet.get("Binds", 0), errors="coerce").max())
                            else:
                                seg_roe = float(pd.to_numeric(seg_row["ROE"], errors="coerce").iloc[0]) if "ROE" in seg_row.columns else np.nan
                                seg_ltv = float(pd.to_numeric(seg_row["Avg. MRLTV"], errors="coerce").iloc[0]) if "Avg. MRLTV" in seg_row.columns else np.nan
                                seg_cpb = float(pd.to_numeric(seg_row["CPB"], errors="coerce").iloc[0]) if "CPB" in seg_row.columns else np.nan
                                seg_binds = float(pd.to_numeric(seg_row["Binds"], errors="coerce").iloc[0]) if "Binds" in seg_row.columns else np.nan

                            ch_rows = rec_df[
                                (rec_df["State"] == state_s)
                                & (rec_df["Segment"] == segment_s)
                                & (rec_df["Channel Groups"] == channel_s)
                            ].copy()
                            ch_bids = float(pd.to_numeric(ch_rows.get("Bids", 0), errors="coerce").fillna(0).sum())
                            ch_clicks = float(pd.to_numeric(ch_rows.get("Clicks", 0), errors="coerce").fillna(0).sum())
                            ch_wr = (ch_clicks / ch_bids) if ch_bids > 0 else np.nan
                            ch_sov = float(pd.to_numeric(ch_rows.get("SOV", np.nan), errors="coerce").mean()) if not ch_rows.empty else np.nan

                            st.markdown("<div class='px-subhead'>KPI Summary</div>", unsafe_allow_html=True)
                            render_kpi_tiles(
                                [
                                    {"label": "State-Segment ROE", "value": "n/a" if pd.isna(seg_roe) else f"{seg_roe:.1%}"},
                                    {"label": "State-Segment MRLTV", "value": "n/a" if pd.isna(seg_ltv) else f"${seg_ltv:,.0f}"},
                                    {"label": "State-Segment CPB", "value": "n/a" if pd.isna(seg_cpb) else f"${seg_cpb:,.0f}"},
                                    {"label": "State-Segment Binds", "value": "n/a" if pd.isna(seg_binds) else f"{seg_binds:,.0f}"},
                                    {"label": "Channel Bids", "value": f"{ch_bids:,.0f}"},
                                    {"label": "Channel Win Rate", "value": "n/a" if pd.isna(ch_wr) else f"{ch_wr:.2%}"},
                                    {"label": "Channel SOV", "value": "n/a" if pd.isna(ch_sov) else f"{ch_sov:.1%}"},
                                ],
                                cols=4,
                            )
                            st.markdown("<div class='px-sep'></div>", unsafe_allow_html=True)
                            st.markdown("<div class='px-subhead'>Testing Impact Chart</div>", unsafe_allow_html=True)
                            sdet = sdet.sort_values("Bid Adj %")
                            bar_df = sdet[[
                                "Adj Label", "Bid Adj %", "Win Rate Uplift", "CPC Uplift",
                                "Sig Icon", "Sig Level", "Source Used", "Evidence Icon", "Evidence Label"
                            ]].copy()
                            bar_df = bar_df.rename(columns={"Win Rate Uplift": "Win-Rate Uplift", "CPC Uplift": "CPC Uplift"})
                            # Decision KPIs always use state+channel volume; fallback only informs uplift deltas.
                            bar_df["State Bids"] = pd.to_numeric(sdet.get("Bids", 0), errors="coerce").fillna(0.0).values
                            bar_df["State Clicks"] = pd.to_numeric(sdet.get("Clicks", 0), errors="coerce").fillna(0.0).values
                            bar_df["Evidence Bids"] = pd.to_numeric(sdet.get("Test Bids", 0), errors="coerce").fillna(0.0).values
                            bar_df["Evidence Clicks"] = pd.to_numeric(sdet.get("Test Clicks", 0), errors="coerce").fillna(0.0).values
                            bar_df["State Bid Share"] = np.where(ch_bids > 0, bar_df["State Bids"] / ch_bids, np.nan)
                            bar_df["State Click Share"] = np.where(ch_clicks > 0, bar_df["State Clicks"] / ch_clicks, np.nan)
                            melted = bar_df.melt(
                                id_vars=[
                                    "Adj Label", "Sig Icon", "Sig Level", "Source Used", "Evidence Icon", "Evidence Label",
                                    "State Bids", "State Clicks", "Evidence Bids", "Evidence Clicks",
                                    "State Bid Share", "State Click Share"
                                ],
                                value_vars=["Win-Rate Uplift", "CPC Uplift"],
                                var_name="Metric",
                                value_name="Change",
                            )
                            fig_px = px.bar(
                                melted,
                                x="Adj Label",
                                y="Change",
                                color="Metric",
                                barmode="group",
                                color_discrete_map={"Win-Rate Uplift": "#22d3ee", "CPC Uplift": "#f59e0b"},
                                template=plotly_template,
                                title="Testing Point Impact: Win-Rate vs CPC",
                            )
                            fig_px.update_traces(
                                texttemplate="%{y:.1%}",
                                textposition="outside",
                                customdata=np.stack(
                                    [
                                        melted["State Bids"].to_numpy(),
                                        melted["State Bid Share"].to_numpy(),
                                        melted["State Clicks"].to_numpy(),
                                        melted["State Click Share"].to_numpy(),
                                        melted["Evidence Bids"].to_numpy(),
                                        melted["Evidence Clicks"].to_numpy(),
                                        melted["Source Used"].to_numpy(),
                                        melted["Sig Level"].to_numpy(),
                                    ],
                                    axis=-1,
                                ),
                                hovertemplate=(
                                    "%{x}<br>"
                                    "%{fullData.name}: %{y:.2%}<br>"
                                    "State Bids: %{customdata[0]:,.0f} (%{customdata[1]:.1%} of total)<br>"
                                    "State Clicks: %{customdata[2]:,.0f} (%{customdata[3]:.1%} of total)<br>"
                                    "Evidence Sample Bids: %{customdata[4]:,.0f}<br>"
                                    "Evidence Sample Clicks: %{customdata[5]:,.0f}<br>"
                                    "Evidence: %{customdata[6]} | Sig: %{customdata[7]}"
                                    "<extra></extra>"
                                ),
                            )
                            y_max = float(max(0.01, np.nanmax(np.abs(melted["Change"].to_numpy(dtype=float)))))
                            icon_df = (
                                bar_df[["Adj Label", "Evidence Icon", "Source Used"]]
                                .drop_duplicates(subset=["Adj Label"])
                                .copy()
                            )
                            icon_df["IconY"] = y_max * 1.18
                            fig_px.add_scatter(
                                x=icon_df["Adj Label"],
                                y=icon_df["IconY"],
                                mode="text",
                                text=icon_df["Evidence Icon"],
                                textposition="middle center",
                                showlegend=False,
                                hovertemplate="Evidence Source: %{customdata[0]}<extra></extra>",
                                customdata=np.stack([icon_df["Source Used"].to_numpy()], axis=-1),
                            )
                            fig_px.update_layout(
                                margin=dict(l=0, r=0, t=48, b=0),
                                yaxis_tickformat=".0%",
                                yaxis_title="Percent Change",
                                xaxis_title="Bid Adjustment Test Point",
                                legend_title_text="Metric",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig_px, use_container_width=True, key=f"tab4_px_chart_{state_s}_{channel_s}_{segment_s}")
                            st.caption("Evidence icons in tooltip: 🟢/🟡 = State+Channel stat-sig, 🔷 = Channel fallback stat-sig.")
                            st.markdown("<div class='px-sep'></div>", unsafe_allow_html=True)
                            st.markdown("<div class='px-subhead'>Testing Point Table</div>", unsafe_allow_html=True)

                            tbl = sdet.copy()
                            tbl["Testing Point"] = tbl["Bid Adj %"].map(lambda x: f"{float(x):+.0f}%")
                            tbl["State Bids"] = tbl["Bids"].map(lambda x: f"{float(x):,.0f}")
                            tbl["Stat-Sig"] = tbl["Sig Icon"].astype(str) + " " + tbl["Sig Level"].astype(str)
                            tbl["Evidence Source"] = tbl["Source Used"].astype(str)
                            tbl["Win Rate Diff"] = tbl["Win Rate Uplift"].map(lambda x: f"{float(x):+.1%}")
                            tbl["Additional Clicks"] = tbl["Additional Clicks"].map(lambda x: f"{float(x):,.0f}")
                            tbl["Additional Binds"] = tbl["Additional Binds"].map(lambda x: f"{float(x):,.2f}")
                            tbl["CPC Uplift"] = tbl["CPC Uplift"].map(lambda x: f"{float(x):+.1%}")
                            tbl["Additional Budget"] = tbl["Additional Budget Needed"].map(lambda x: f"${float(x):,.0f}")
                            show_tbl = tbl[
                                [
                                    "Testing Point",
                                    "State Bids",
                                    "Stat-Sig",
                                    "Evidence Source",
                                    "Win Rate Diff",
                                    "Additional Clicks",
                                    "Additional Binds",
                                    "CPC Uplift",
                                    "Additional Budget",
                                ]
                            ]
                            st.dataframe(show_tbl, use_container_width=True, hide_index=True)

    elif selected_tab == tab_labels[5]:
        st.subheader("📚 General Analytics")
        st.caption("Drag dimensions to row groups/pivot, reorder, and hide columns from the Columns panel.")
        st.markdown(
            """
            <style>
            .ga-shell {
                border: 1px solid rgba(45,212,191,0.62);
                border-radius: 14px;
                box-shadow: 0 0 0 1px rgba(45,212,191,0.18), 0 8px 24px rgba(2,6,23,0.35);
                background: linear-gradient(145deg, rgba(10,16,28,0.64), rgba(10,16,28,0.38));
                padding: 8px 10px 10px 10px;
                margin-top: 8px;
            }
            .ga-note {
                color: #93c5fd;
                font-size: 0.82rem;
                margin-bottom: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        _master_a, detail_a = build_price_exploration_master_detail(rec_df, price_eval, settings)
        analytics_df = build_general_analytics_df(rec_df, state_df, detail_a)
        if analytics_df.empty:
            st.info("No analytics rows available for current filters/data.")
        elif AGGRID_AVAILABLE:
            dim_cols = ["Product Strategy", "State", "Channel Groups", "Testing Point"]
            metric_cols = [
                "Num Bids",
                "Num Impressions",
                "Avg Bid",
                "CPC",
                "Win Rate",
                "Cost",
                "Channel QSR",
                "Channel Q2C",
                "Channel Clicks",
                "Channel Quotes",
                "State Binds",
                "State ROE",
                "State Combined Ratio",
            ]
            show_cols = [c for c in dim_cols + metric_cols + ["Source Used"] if c in analytics_df.columns]
            gdf = analytics_df[show_cols].copy()

            if "tab5_presets" not in st.session_state:
                st.session_state["tab5_presets"] = load_analytics_presets()
            presets = st.session_state.get("tab5_presets", {})
            default_preset_name = str(presets.get("__default__", "") or "")
            preset_names = sorted([str(k) for k in presets.keys() if not str(k).startswith("__")])
            if (
                st.session_state.get("tab5_preset_select") in {None, ""}
                and default_preset_name in preset_names
            ):
                st.session_state["tab5_preset_select"] = default_preset_name
            st.markdown("**Presets**")
            p1, p2, p4, p5, p6, p7 = st.columns([1.2, 1.45, 1.05, 1.2, 0.9, 0.85])
            selected_preset = p1.selectbox(
                "Preset",
                options=["(none)"] + preset_names,
                key="tab5_preset_select",
            )
            preset_name_input = p2.text_input("Preset name", value=selected_preset if selected_preset != "(none)" else "", key="tab5_preset_name")
            save_as_clicked = p4.button("💾 Save As", key="tab5_save_as_preset_btn")
            update_clicked = p5.button("🔄 Update Preset", key="tab5_update_preset_btn") if selected_preset != "(none)" else False
            set_default_clicked = p6.button("⭐ Default", key="tab5_set_default_btn", disabled=(selected_preset == "(none)"))
            delete_clicked = p7.button("🗑", key="tab5_delete_preset_btn", disabled=(selected_preset == "(none)"))
            if default_preset_name:
                st.caption(f"Default preset: `{default_preset_name}`")

            colorable_cols = [c for c in metric_cols if c in gdf.columns]
            default_color_cols = st.session_state.get("tab5_color_cols", ["State Combined Ratio"] if "State Combined Ratio" in colorable_cols else [])
            color_cols = st.multiselect(
                "Color-code columns",
                options=colorable_cols,
                default=[c for c in default_color_cols if c in colorable_cols],
                key="tab5_color_cols",
                help="Select metric columns for gradient color scale.",
            )

            loaded_preset = presets.get(selected_preset, {}) if selected_preset != "(none)" else {}

            gb = GridOptionsBuilder.from_dataframe(gdf)
            gb.configure_default_column(
                resizable=True,
                sortable=True,
                filter=True,
                enableRowGroup=True,
                enablePivot=True,
                enableValue=True,
            )
            for c in dim_cols:
                if c in gdf.columns:
                    gb.configure_column(
                        c,
                        rowGroup=(c in ["Product Strategy", "State", "Channel Groups", "Testing Point"]),
                        hide=(c in ["Product Strategy", "State", "Channel Groups", "Testing Point"]),
                    )
            for c in metric_cols:
                if c in gdf.columns:
                    agg = "sum" if c in ["Num Bids", "Num Impressions", "Cost", "State Binds"] else "avg"
                    gb.configure_column(c, type=["numericColumn"], aggFunc=agg)
                    if c in color_cols:
                        cmin = float(pd.to_numeric(gdf[c], errors="coerce").min()) if pd.to_numeric(gdf[c], errors="coerce").notna().any() else 0.0
                        cmax = float(pd.to_numeric(gdf[c], errors="coerce").max()) if pd.to_numeric(gdf[c], errors="coerce").notna().any() else 1.0
                        if c == "State Combined Ratio":
                            style_js = JsCode(
                                """
                                function(p){
                                    const v = Number(p.value);
                                    if(!Number.isFinite(v)) return {};
                                    function lerp(a,b,t){return Math.round(a + (b-a)*t);}
                                    function mix(c1,c2,t){return `rgb(${lerp(c1[0],c2[0],t)},${lerp(c1[1],c2[1],t)},${lerp(c1[2],c2[2],t)})`;}
                                    let bg = '#8ab4f8';
                                    let fg = '#0b1020';
                                    if(v >= 1.15){
                                        const t = Math.min((v-1.15)/0.20, 1.0);
                                        bg = mix([244,67,54],[183,28,28],t);   // vivid red gradient
                                        fg = '#ffffff';
                                    }else if(v >= 1.05){
                                        const t = (v-1.05)/0.10;
                                        bg = mix([255,202,40],[251,140,0],t);  // vivid amber gradient
                                        fg = '#0b1020';
                                    }else if(v >= 1.00){
                                        const t = (v-1.00)/0.05;
                                        bg = mix([174,213,129],[102,187,106],t); // light green gradient
                                        fg = '#0b1020';
                                    }else{
                                        const t = Math.min((1.00-v)/0.15, 1.0);
                                        bg = mix([102,187,106],[46,125,50],t); // strong green gradient
                                        fg = '#ffffff';
                                    }
                                    return {'backgroundColor': bg, 'color':fg, 'fontWeight':'700'};
                                }
                                """
                            )
                        else:
                            # No explicit thresholds: use min->max gradient.
                            style_js = JsCode(
                                f"""
                                function(p){{
                                    const v = Number(p.value);
                                    if(!Number.isFinite(v)) return {{}};
                                    const min = {cmin};
                                    const max = {cmax if cmax != cmin else cmin + 1.0};
                                    const tRaw = (v - min) / (max - min);
                                    const t = Math.max(0, Math.min(1, tRaw));
                                    // Google-style vivid 3-stop gradient: red -> yellow -> green.
                                    function lerp(a,b,x){{ return Math.round(a + (b-a)*x); }}
                                    function mix(c1,c2,x){{ return [lerp(c1[0],c2[0],x), lerp(c1[1],c2[1],x), lerp(c1[2],c2[2],x)]; }}
                                    const red = [234,67,53];
                                    const yellow = [251,188,5];
                                    const green = [52,168,83];
                                    let rgb = [0,0,0];
                                    if(t <= 0.5){{
                                        rgb = mix(red, yellow, t / 0.5);
                                    }} else {{
                                        rgb = mix(yellow, green, (t - 0.5) / 0.5);
                                    }}
                                    const luminance = (0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]) / 255;
                                    const fg = luminance > 0.62 ? '#0b1020' : '#ffffff';
                                    return {{'backgroundColor': `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`, 'color':fg, 'fontWeight':'700'}};
                                }}
                                """
                            )
                        gb.configure_column(c, cellStyle=style_js)

            int_fmt = JsCode("function(p){const v=Number(p.value);return Number.isFinite(v)?Math.round(v).toLocaleString():'';}")
            pct0_fmt = JsCode("function(p){const v=Number(p.value);return Number.isFinite(v)?Math.round(v*100).toLocaleString()+'%':'';}")
            usd0_fmt = JsCode("function(p){const v=Number(p.value);return Number.isFinite(v)?('$'+Math.round(v).toLocaleString()):'';}")

            for c in ["Num Bids", "Num Impressions", "Channel Clicks", "Channel Quotes", "State Binds"]:
                if c in gdf.columns:
                    gb.configure_column(c, valueFormatter=int_fmt)
            for c in ["Win Rate", "Channel QSR", "Channel Q2C", "State ROE", "State Combined Ratio"]:
                if c in gdf.columns:
                    gb.configure_column(c, valueFormatter=pct0_fmt)
            for c in ["Avg Bid", "CPC", "Cost"]:
                if c in gdf.columns:
                    gb.configure_column(c, valueFormatter=usd0_fmt)

            go = gb.build()
            go["rowGroupPanelShow"] = "always"
            go["pivotPanelShow"] = "always"
            go["animateRows"] = True
            go["sideBar"] = {"toolPanels": ["columns", "filters"], "defaultToolPanel": "columns"}
            go["groupDisplayType"] = "singleColumn"
            go["autoGroupColumnDef"] = {
                "headerName": "Drilldown",
                "minWidth": 280,
                "cellRendererParams": {"suppressCount": False},
            }
            go = apply_grid_preset(go, loaded_preset)
            expand_mode = st.session_state.get("tab5_expand_mode", "collapsed")
            go["groupDefaultExpanded"] = 99 if expand_mode == "expanded" else 0
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("Expand all", key="tab5_expand_all_btn"):
                st.session_state["tab5_expand_mode"] = "expanded"
                st.rerun()
            if b2.button("Collapse all", key="tab5_collapse_all_btn"):
                st.session_state["tab5_expand_mode"] = "collapsed"
                st.rerun()

            custom_css = {
                ".ag-root-wrapper": {
                    "border": "1px solid rgba(45,212,191,0.45)",
                    "border-radius": "10px",
                    "background-color": "#081225",
                },
                ".ag-root-wrapper-body": {"background-color": "#081225"},
                ".ag-header": {"background-color": "#0b1730", "border-bottom": "1px solid rgba(45,212,191,0.20)"},
                ".ag-header-cell": {"background-color": "#0b1730"},
                ".ag-header-cell-label": {"color": "#e2e8f0", "font-weight": "700"},
                ".ag-cell": {"color": "#dbeafe", "background-color": "#081225", "border-color": "rgba(148,163,184,0.12)"},
                ".ag-row": {"background-color": "#081225"},
                ".ag-row-even": {"background-color": "#09172b"},
                ".ag-row-hover": {"background-color": "rgba(34,211,238,0.10) !important"},
                ".ag-row-selected": {"background-color": "rgba(34,211,238,0.16) !important"},
                ".ag-row-group": {"color": "#f8fafc", "font-weight": "700"},
                ".ag-body-viewport": {"background-color": "#081225"},
                ".ag-center-cols-viewport": {"background-color": "#081225"},
                ".ag-center-cols-container": {"background-color": "#081225"},
                ".ag-body-horizontal-scroll": {"background-color": "#0a1428"},
                ".ag-body-vertical-scroll": {"background-color": "#0a1428"},
                ".ag-body-horizontal-scroll-viewport": {"background-color": "#0a1428"},
                ".ag-body-vertical-scroll-viewport": {"background-color": "#0a1428"},
                ".ag-input-field-input": {"background-color": "#0b1730", "color": "#e2e8f0", "border": "1px solid rgba(45,212,191,0.28)"},
                ".ag-text-field-input": {"background-color": "#0b1730", "color": "#e2e8f0", "border": "1px solid rgba(45,212,191,0.28)"},
                "input[class*='ag-']": {"background-color": "#0b1730", "color": "#e2e8f0"},
                ".ag-side-bar": {"background-color": "#081225", "border-left": "1px solid rgba(45,212,191,0.18)"},
                ".ag-tool-panel-wrapper": {
                    "background-color": "#081225",
                    "color": "#cbd5e1",
                    "display": "flex",
                    "flex-direction": "column",
                    "justify-content": "flex-start !important",
                    "align-items": "stretch",
                },
                ".ag-tool-panel-wrapper *": {"color": "#cbd5e1"},
                ".ag-column-select-panel": {
                    "display": "flex",
                    "flex-direction": "column",
                    "justify-content": "flex-start !important",
                    "align-items": "stretch",
                },
                ".ag-column-select-panel .ag-column-drop": {
                    "margin-top": "0 !important",
                    "margin-bottom": "8px !important",
                    "padding-top": "0 !important",
                    "align-self": "stretch !important",
                },
                ".ag-column-drop-vertical": {
                    "display": "flex",
                    "flex-direction": "column",
                    "align-items": "flex-start !important",
                    "justify-content": "flex-start !important",
                },
                ".ag-column-drop-vertical .ag-column-drop-list": {
                    "display": "flex",
                    "flex-direction": "column",
                    "align-content": "flex-start !important",
                    "justify-content": "flex-start !important",
                    "align-items": "flex-start !important",
                    "min-height": "auto !important",
                    "height": "auto !important",
                    "padding-top": "2px !important",
                },
                ".ag-column-drop-empty-message": {
                    "margin-top": "4px !important",
                    "padding-top": "0 !important",
                    "align-self": "flex-start !important",
                },
                ".ag-column-drop-vertical .ag-column-drop-cell": {"margin-top": "4px"},
                ".ag-checkbox-input-wrapper": {"opacity": "1 !important"},
                ".ag-icon": {"color": "#cbd5e1 !important"},
                ".ag-paging-panel": {"background-color": "#081225", "color": "#cbd5e1"},
            }
            st.markdown("<div class='ga-shell'><div class='ga-note'>Drag dimensions in Columns panel to Row Groups / Columns / Values.</div>", unsafe_allow_html=True)
            # Taller default viewport to show ~100 rows as requested.
            grid_height = 3000
            grid_resp = AgGrid(
                gdf,
                gridOptions=go,
                fit_columns_on_grid_load=False,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                height=grid_height,
                enable_enterprise_modules=True,
                theme="balham-dark" if dark_mode else "balham",
                custom_css=custom_css,
                allow_unsafe_jscode=True,
                key=tab5_grid_component_key(selected_preset),
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if isinstance(grid_resp, dict):
                captured_state = {
                    "column_state": grid_resp.get("column_state") or grid_resp.get("columns_state") or grid_resp.get("columnState"),
                    "filter_model": grid_resp.get("filter_model") or grid_resp.get("filterModel"),
                    "sort_model": grid_resp.get("sort_model") or grid_resp.get("sortModel"),
                    "pivot_mode": grid_resp.get("pivot_mode") if "pivot_mode" in grid_resp else None,
                    "grid_state": grid_resp.get("grid_state") or grid_resp.get("gridState"),
                }
                if any(v is not None and v != {} and v != [] for v in captured_state.values()):
                    st.session_state["tab5_last_grid_state"] = captured_state

            if save_as_clicked:
                nm = str(preset_name_input or "").strip()
                if not nm:
                    st.warning("Please enter a preset name.")
                else:
                    payload = st.session_state.get("tab5_last_grid_state", {}) or {}
                    presets = dict(st.session_state.get("tab5_presets", {}))
                    presets[nm] = payload
                    okp, errp = save_analytics_presets(presets)
                    if okp:
                        st.session_state["tab5_presets"] = presets
                        st.success(f"Preset saved: {nm}")
                    else:
                        st.error(errp)

            if update_clicked and selected_preset != "(none)":
                payload = st.session_state.get("tab5_last_grid_state", {}) or {}
                presets = dict(st.session_state.get("tab5_presets", {}))
                presets[selected_preset] = payload
                okp2, errp2 = save_analytics_presets(presets)
                if okp2:
                    st.session_state["tab5_presets"] = presets
                    st.success(f"Preset updated: {selected_preset}")
                else:
                    st.error(errp2)

            if set_default_clicked and selected_preset != "(none)":
                presets = dict(st.session_state.get("tab5_presets", {}))
                presets["__default__"] = selected_preset
                okd, errd = save_analytics_presets(presets)
                if okd:
                    st.session_state["tab5_presets"] = presets
                    st.success(f"Default preset set: {selected_preset}")
                else:
                    st.error(errd)

            if delete_clicked and selected_preset != "(none)":
                presets = dict(st.session_state.get("tab5_presets", {}))
                presets.pop(selected_preset, None)
                if presets.get("__default__", "") == selected_preset:
                    presets["__default__"] = ""
                okx, errx = save_analytics_presets(presets)
                if okx:
                    st.session_state["tab5_presets"] = presets
                    st.session_state["tab5_preset_select"] = "(none)"
                    st.success(f"Preset deleted: {selected_preset}")
                    st.rerun()
                else:
                    st.error(errx)
        else:
            st.info("`streamlit-aggrid` is unavailable in this environment. Showing static table fallback.")
            render_formatted_table(analytics_df, use_container_width=True)

    elif selected_tab == tab_labels[6]:
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
