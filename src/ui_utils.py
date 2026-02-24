import numpy as np
import pandas as pd
import streamlit as st

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

