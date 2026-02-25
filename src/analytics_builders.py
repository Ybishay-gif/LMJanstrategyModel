import numpy as np
import pandas as pd
import streamlit as st

from config import Settings, effective_cpc_cap_pct, mode_factor


def _build_state_point_effects_for_detail(price_eval_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    pe = price_eval_df.copy()
    if pe.empty or "State" not in pe.columns:
        return pd.DataFrame()

    st = (
        pe.groupby(["State", "Channel Groups", "Price Adjustment Percent"], as_index=False)
        .agg(
            Test_Bids=("Bids", "sum"),
            Test_Clicks=("Clicks", "sum"),
            **{"State Win Rate Uplift": ("Win Rate Lift %", "mean")},
            **{"State CPC Uplift": ("CPC Lift %", "mean")},
            **{"State Sig": ("Stat Sig Price Point", "max")},
        )
    )
    ch = (
        pe.groupby(["Channel Groups", "Price Adjustment Percent"], as_index=False)
        .agg(
            **{"Channel Test Bids": ("Bids", "sum")},
            **{"Channel Test Clicks": ("Clicks", "sum")},
            **{"Channel Win Rate Uplift": ("Win Rate Lift %", "mean")},
            **{"Channel CPC Uplift": ("CPC Lift %", "mean")},
            **{"Channel Sig": ("Stat Sig Price Point", "max")},
        )
    )

    out = st.merge(ch, on=["Channel Groups", "Price Adjustment Percent"], how="left")
    out["State Sig"] = out["State Sig"].fillna(False).astype(bool)
    out["Channel Sig"] = out["Channel Sig"].fillna(False).astype(bool)

    use_state = out["State Sig"]
    use_channel = (~use_state) & out["Channel Sig"]
    out["Source Used"] = np.select(
        [use_state, use_channel],
        ["State+Channel", "Channel Fallback"],
        default="No Sig",
    )
    out["Recommend_Eligible"] = use_state | use_channel
    out["Win Rate Uplift"] = np.where(
        use_state,
        out["State Win Rate Uplift"],
        np.where(use_channel, out["Channel Win Rate Uplift"], np.nan),
    )
    out["CPC Uplift"] = np.where(
        use_state,
        out["State CPC Uplift"],
        np.where(use_channel, out["Channel CPC Uplift"], np.nan),
    )
    out["Evidence Bids"] = np.where(use_state, out["Test_Bids"], np.where(use_channel, out["Channel Test Bids"], out["Test_Bids"]))
    out["Evidence Clicks"] = np.where(use_state, out["Test_Clicks"], np.where(use_channel, out["Channel Test Clicks"], out["Test_Clicks"]))

    bids_sig = max(float(settings.min_bids_price_sig), 1.0)
    clicks_sig = max(float(settings.min_clicks_price_sig), 1.0)
    out["Sig_Score"] = np.minimum(
        pd.to_numeric(out["Evidence Bids"], errors="coerce").fillna(0.0) / bids_sig,
        pd.to_numeric(out["Evidence Clicks"], errors="coerce").fillna(0.0) / clicks_sig,
    )
    out["Sig Level"] = np.select(
        [
            out["Recommend_Eligible"] & (out["Sig_Score"] >= 2.5),
            out["Recommend_Eligible"] & (out["Sig_Score"] >= 1.5),
            out["Recommend_Eligible"],
            (pd.to_numeric(out["Evidence Bids"], errors="coerce").fillna(0.0) > 0)
            | (pd.to_numeric(out["Evidence Clicks"], errors="coerce").fillna(0.0) > 0),
        ],
        ["Strong", "Medium", "Weak", "Poor"],
        default="No Sig",
    )
    out["Sig Icon"] = out["Sig Level"].map(
        {"Strong": "🟢", "Medium": "🟡", "Weak": "🟠", "Poor": "🟠", "No Sig": "⚪"}
    ).fillna("⚪")
    return out


def build_price_exploration_master_detail(
    rec_df: pd.DataFrame,
    price_eval_df: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rec_df is None or rec_df.empty or price_eval_df is None or price_eval_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = rec_df.groupby(["State", "Channel Groups", "Segment"], as_index=False).agg(
        Bids=("Bids", "sum"),
        Clicks=("Clicks", "sum"),
        Binds=("Binds", "sum"),
        **{"Avg. CPC": ("Avg. CPC", "mean")},
        **{"Current Cost": ("Total Click Cost", "sum")},
        **{"Row Win Rate": ("Bids to Clicks", "mean")},
        **{"Row C2B": ("Clicks to Binds", "mean")},
    )
    base["SC Key"] = base["State"].astype(str) + "|" + base["Channel Groups"].astype(str)

    cand = _build_state_point_effects_for_detail(price_eval_df, settings)
    if cand.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail_keys = base[["State", "Channel Groups", "Segment", "SC Key"]].drop_duplicates().merge(
        cand,
        on=["State", "Channel Groups"],
        how="left",
    )
    detail_keys = detail_keys[detail_keys["Price Adjustment Percent"].notna()].copy()

    if detail_keys.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail = detail_keys.merge(
        base[
            [
                "State", "Channel Groups", "Segment", "Bids", "Clicks", "Binds",
                "Avg. CPC", "Current Cost", "Row Win Rate", "Row C2B",
            ]
        ],
        on=["State", "Channel Groups", "Segment"],
        how="left",
    )
    wr_lift = detail["Win Rate Uplift"].fillna(0.0)
    allow_negative_moves = mode_factor(settings.optimization_mode) <= 0.5
    wr_effect = wr_lift if allow_negative_moves else np.maximum(wr_lift, 0.0)
    detail["Additional Clicks"] = detail["Bids"].fillna(0.0) * detail["Row Win Rate"].fillna(0.0) * wr_effect
    detail["Additional Binds"] = detail["Additional Clicks"] * detail["Row C2B"].fillna(0.0)
    detail["Expected Total Cost"] = (
        (detail["Clicks"].fillna(0.0) + detail["Additional Clicks"])
        * detail["Avg. CPC"].fillna(0.0)
        * (1.0 + detail["CPC Uplift"].fillna(0.0))
    )
    detail["Additional Budget Needed"] = detail["Expected Total Cost"] - detail["Current Cost"].fillna(0.0)
    detail["Adj Label"] = detail["Price Adjustment Percent"].map(lambda x: f"{float(x):+.0f}%")
    detail["Evidence Icon"] = np.select(
        [
            detail["Source Used"].eq("State+Channel"),
            detail["Source Used"].eq("Channel Fallback"),
            detail["Source Used"].eq("Channel"),
        ],
        [
            np.where(detail["Sig Level"].eq("Strong"), "🟢", "🟡"),
            "🔷",
            "🔹",
        ],
        default="⚪",
    )
    detail["Evidence Label"] = np.select(
        [
            detail["Source Used"].eq("State+Channel"),
            detail["Source Used"].eq("Channel Fallback"),
            detail["Source Used"].eq("Channel"),
        ],
        [
            detail["Sig Level"].astype(str) + " (State+Channel)",
            detail["Sig Level"].astype(str) + " (Channel Fallback)",
            detail["Sig Level"].astype(str) + " (Channel Only)",
        ],
        default=detail["Sig Level"].astype(str) + " (Unknown Source)",
    )
    detail = detail.rename(
        columns={
            "Price Adjustment Percent": "Bid Adj %",
            "Test_Bids": "Test Bids",
            "Test_Clicks": "Test Clicks",
        }
    )
    detail = detail.sort_values(["State", "Channel Groups", "Segment", "Bid Adj %"]).reset_index(drop=True)

    master = detail.groupby(["State", "Channel Groups", "Segment"], as_index=False).agg(
        **{"Total Bids": ("Bids", "max")},
        **{"Total Clicks": ("Clicks", "max")},
        **{"Total Binds": ("Binds", "max")},
        **{"Testing Points Count": ("Bid Adj %", "count")},
        **{"Source Used": ("Source Used", "first")},
    )
    points_map = (
        detail.groupby(["State", "Channel Groups", "Segment"])["Adj Label"]
        .apply(lambda s: " || ".join(s.tolist()))
        .reset_index(name="Testing Points")
    )
    master = master.merge(points_map, on=["State", "Channel Groups", "Segment"], how="left")
    return master, detail


@st.cache_data(show_spinner=False)
def build_price_exploration_detail_lookup(detail_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if detail_df is None or detail_df.empty:
        return {}
    out: dict[str, pd.DataFrame] = {}
    gcols = ["State", "Channel Groups", "Segment"]
    for (state, channel_group, segment), g in detail_df.groupby(gcols, dropna=False):
        k = f"{state}|{channel_group}|{segment}"
        out[k] = g.sort_values("Bid Adj %").reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def build_general_analytics_df(
    rec_df: pd.DataFrame,
    state_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> pd.DataFrame:
    if rec_df is None or rec_df.empty or detail_df is None or detail_df.empty:
        return pd.DataFrame()

    base = rec_df.groupby(["State", "Channel Groups"], as_index=False).agg(
        **{"Product Strategy": ("Strategy Bucket", "first")},
        **{"Num Bids": ("Bids", "sum")},
        **{"Num Impressions": ("Impressions", "sum")},
        **{"Clicks (State-Channel)": ("Clicks", "sum")},
        **{"Quotes (State-Channel)": ("Quotes", "sum")},
        **{"Cost": ("Total Click Cost", "sum")},
    )
    base["Cost"] = np.where(
        pd.to_numeric(base["Cost"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(base["Cost"], errors="coerce").fillna(0),
        0.0,
    )

    w = rec_df.copy()
    w["__w_bids__"] = pd.to_numeric(w["Bids"], errors="coerce").fillna(0.0)
    w["__avg_bid_x_w__"] = pd.to_numeric(w["Avg. Bid"], errors="coerce").fillna(0.0) * w["__w_bids__"]
    w["__avg_cpc_x_clicks__"] = pd.to_numeric(w["Avg. CPC"], errors="coerce").fillna(0.0) * pd.to_numeric(w["Clicks"], errors="coerce").fillna(0.0)
    w["__wr_x_bids__"] = pd.to_numeric(w["Bids to Clicks"], errors="coerce").fillna(0.0) * w["__w_bids__"]
    wt = w.groupby(["State", "Channel Groups"], as_index=False).agg(
        __w_bids__=("__w_bids__", "sum"),
        __clicks__=("Clicks", "sum"),
        __avg_bid_x_w__=("__avg_bid_x_w__", "sum"),
        __avg_cpc_x_clicks__=("__avg_cpc_x_clicks__", "sum"),
        __wr_x_bids__=("__wr_x_bids__", "sum"),
    )
    wt["Avg Bid"] = np.where(wt["__w_bids__"] > 0, wt["__avg_bid_x_w__"] / wt["__w_bids__"], np.nan)
    wt["CPC"] = np.where(wt["__clicks__"] > 0, wt["__avg_cpc_x_clicks__"] / wt["__clicks__"], np.nan)
    wt["Win Rate"] = np.where(wt["__w_bids__"] > 0, wt["__wr_x_bids__"] / wt["__w_bids__"], np.nan)
    wt = wt[["State", "Channel Groups", "Avg Bid", "CPC", "Win Rate"]]
    base = base.merge(wt, on=["State", "Channel Groups"], how="left")

    ch_all = rec_df.groupby("Channel Groups", as_index=False).agg(
        __ch_clicks=("Clicks", "sum"),
        __ch_quotes=("Quotes", "sum"),
        __ch_qsr_x_clicks=("Quote Start Rate", lambda s: np.nansum(pd.to_numeric(s, errors="coerce") * pd.to_numeric(rec_df.loc[s.index, "Clicks"], errors="coerce"))),
        __ch_q2c_x_clicks=("Clicks to Quotes", lambda s: np.nansum(pd.to_numeric(s, errors="coerce") * pd.to_numeric(rec_df.loc[s.index, "Clicks"], errors="coerce"))),
    )
    ch_all["Channel Clicks"] = ch_all["__ch_clicks"]
    ch_all["Channel Quotes"] = ch_all["__ch_quotes"]
    ch_all["Channel QSR"] = np.where(ch_all["__ch_clicks"] > 0, ch_all["__ch_qsr_x_clicks"] / ch_all["__ch_clicks"], np.nan)
    ch_all["Channel Q2C"] = np.where(ch_all["__ch_clicks"] > 0, ch_all["__ch_q2c_x_clicks"] / ch_all["__ch_clicks"], np.nan)
    ch_all = ch_all[["Channel Groups", "Channel QSR", "Channel Q2C", "Channel Clicks", "Channel Quotes"]]

    st_all = state_df[["State", "Binds", "ROE", "Combined Ratio"]].copy()
    st_all = st_all.rename(
        columns={
            "Binds": "State Binds",
            "ROE": "State ROE",
            "Combined Ratio": "State Combined Ratio",
        }
    )

    d = detail_df.copy()
    d["Testing Point"] = d["Bid Adj %"].map(lambda x: f"{float(x):+.0f}%")
    d["__src_rank__"] = np.select(
        [d["Source Used"].eq("State+Channel"), d["Source Used"].eq("Channel Fallback"), d["Source Used"].eq("Channel")],
        [0, 1, 2],
        default=3,
    )
    d = d.sort_values(["State", "Channel Groups", "Bid Adj %", "__src_rank__"])
    d = d.drop_duplicates(subset=["State", "Channel Groups", "Bid Adj %"], keep="first")
    d = d[["State", "Channel Groups", "Testing Point", "Bid Adj %", "Source Used"]]

    out = d.merge(base, on=["State", "Channel Groups"], how="left")
    out = out.merge(ch_all, on="Channel Groups", how="left")
    out = out.merge(st_all, on="State", how="left")

    if not out.empty:
        sc_tp_cnt = (
            out.groupby(["State", "Channel Groups"], dropna=False)["Testing Point"]
            .transform("nunique")
            .replace(0, 1)
            .fillna(1.0)
        )
        for c in ["Num Bids", "Num Impressions", "Clicks (State-Channel)", "Quotes (State-Channel)", "Cost"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0) / sc_tp_cnt

        st_row_cnt = (
            out.groupby(["State"], dropna=False)["State"]
            .transform("count")
            .replace(0, 1)
            .fillna(1.0)
        )
        if "State Binds" in out.columns:
            out["State Binds"] = pd.to_numeric(out["State Binds"], errors="coerce").fillna(0.0) / st_row_cnt
    return out
