import numpy as np
import pandas as pd
import streamlit as st

from config import Settings, effective_cpc_cap_pct, mode_factor


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

    pe = price_eval_df.copy()
    pe = pe[
        pe["Stat Sig Price Point"].fillna(False)
        & (pe["CPC Lift %"].fillna(0) <= effective_cpc_cap_pct(settings) / 100.0)
    ].copy()
    min_adj = -30.0 if mode_factor(settings.optimization_mode) <= 0.5 else 0.0
    pe = pe[pe["Price Adjustment Percent"].fillna(0) >= min_adj].copy()
    if pe.empty:
        return pd.DataFrame(), pd.DataFrame()

    bids_sig = max(float(settings.min_bids_price_sig), 1.0)
    clicks_sig = max(float(settings.min_clicks_price_sig), 1.0)
    pe["Sig Score"] = np.minimum(pe["Bids"].fillna(0) / bids_sig, pe["Clicks"].fillna(0) / clicks_sig)
    pe["Sig Level"] = np.select(
        [pe["Sig Score"] >= 2.5, pe["Sig Score"] >= 1.5],
        ["Strong", "Medium"],
        default="Weak",
    )
    pe = pe[pe["Sig Level"] != "Weak"].copy()
    if pe.empty:
        return pd.DataFrame(), pd.DataFrame()
    pe["Sig Icon"] = pe["Sig Level"].map({"Strong": "🟢", "Medium": "🟡"}).fillna("⚪")

    grp_cols = ["Channel Groups", "Price Adjustment Percent"]
    if "State" in pe.columns:
        grp_cols = ["State", "Channel Groups", "Price Adjustment Percent"]

    cand_state = pe.groupby(grp_cols, as_index=False).agg(
        Test_Bids=("Bids", "sum"),
        Test_Clicks=("Clicks", "sum"),
        **{"Win Rate Uplift": ("Win Rate Lift %", "mean")},
        **{"CPC Uplift": ("CPC Lift %", "mean")},
        Sig_Score=("Sig Score", "mean"),
    )
    cand_state["Sig Level"] = np.select(
        [cand_state["Sig_Score"] >= 2.5, cand_state["Sig_Score"] >= 1.5],
        ["Strong", "Medium"],
        default="Weak",
    )
    cand_state = cand_state[cand_state["Sig Level"] != "Weak"].copy()
    cand_state["Sig Icon"] = cand_state["Sig Level"].map({"Strong": "🟢", "Medium": "🟡"}).fillna("⚪")

    if "State" in cand_state.columns:
        cand_ch = cand_state.groupby(["Channel Groups", "Price Adjustment Percent"], as_index=False).agg(
            Test_Bids=("Test_Bids", "sum"),
            Test_Clicks=("Test_Clicks", "sum"),
            **{"Win Rate Uplift": ("Win Rate Uplift", "mean")},
            **{"CPC Uplift": ("CPC Uplift", "mean")},
            Sig_Score=("Sig_Score", "mean"),
        )
        cand_ch["Sig Level"] = np.select(
            [cand_ch["Sig_Score"] >= 2.5, cand_ch["Sig_Score"] >= 1.5],
            ["Strong", "Medium"],
            default="Weak",
        )
        cand_ch = cand_ch[cand_ch["Sig Level"] != "Weak"].copy()
        cand_ch["Sig Icon"] = cand_ch["Sig Level"].map({"Strong": "🟢", "Medium": "🟡"}).fillna("⚪")

        state_pairs = base[["State", "Channel Groups", "Segment", "SC Key"]].drop_duplicates()
        merged_parts: list[pd.DataFrame] = []
        for _, pr in state_pairs.iterrows():
            st = pr["State"]
            ch = pr["Channel Groups"]
            sg = pr["Segment"]
            sk = pr["SC Key"]
            st_rows = cand_state[(cand_state["State"] == st) & (cand_state["Channel Groups"] == ch)].copy()
            ch_rows = cand_ch[cand_ch["Channel Groups"] == ch].copy()

            if st_rows.empty and ch_rows.empty:
                continue

            parts: list[pd.DataFrame] = []
            if not st_rows.empty:
                st_rows["Source Used"] = "State+Channel"
                parts.append(st_rows)
            if not ch_rows.empty:
                st_adj = set(pd.to_numeric(st_rows.get("Price Adjustment Percent", pd.Series(dtype=float)), errors="coerce").dropna().astype(float).tolist())
                ch_rows["Source Used"] = "Channel Fallback"
                ch_rows = ch_rows[
                    ~pd.to_numeric(ch_rows["Price Adjustment Percent"], errors="coerce").astype(float).isin(st_adj)
                ].copy()
                if not ch_rows.empty:
                    parts.append(ch_rows)
            if not parts:
                continue
            comb = pd.concat(parts, ignore_index=True)
            comb["State"] = st
            comb["Channel Groups"] = ch
            comb["Segment"] = sg
            comb["SC Key"] = sk
            merged_parts.append(comb)
        detail_keys = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame()
    else:
        detail_keys = base[["State", "Channel Groups", "Segment", "SC Key"]].drop_duplicates().merge(
            cand_state,
            on=["Channel Groups"],
            how="left",
        )
        detail_keys = detail_keys[detail_keys["Price Adjustment Percent"].notna()].copy()
        detail_keys["Source Used"] = "Channel"

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
