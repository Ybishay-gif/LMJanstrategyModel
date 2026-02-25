import re
from typing import Optional

import numpy as np
import pandas as pd


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


def format_adj_option_label(adj: float, click_uplift: float, cpc_uplift: float, cpb_impact: float, sig_level: str) -> str:
    cpb_txt = "n/a" if pd.isna(cpb_impact) else f"{cpb_impact:+.1%}"
    return (
        f"{adj:+.0f}%: {click_uplift:+.1%} Clicks || {cpc_uplift:+.1%} CPC || {cpb_txt} CPB "
        f"({str(sig_level).lower()} stat-sig)"
    )


def parse_adj_from_label(label: str) -> Optional[float]:
    if not isinstance(label, str):
        return None
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", label)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def apply_grid_preset(go: dict, loaded_preset: Optional[dict]) -> dict:
    if not isinstance(go, dict):
        go = {}
    if not isinstance(loaded_preset, dict):
        return go
    cs = loaded_preset.get("column_state")
    fm = loaded_preset.get("filter_model")
    sm = loaded_preset.get("sort_model")
    pm = loaded_preset.get("pivot_mode")
    gs = loaded_preset.get("grid_state")
    if cs:
        go["columnState"] = cs
    if fm:
        go["filterModel"] = fm
    if sm:
        go["sortModel"] = sm
    if pm is not None:
        go["pivotMode"] = bool(pm)
    if gs:
        go["initialState"] = gs
    return go


def tab5_grid_component_key(selected_preset: str) -> str:
    p = str(selected_preset or "(none)").strip() or "(none)"
    return f"tab5_general_analytics_grid::{p}"


def as_float(v, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def close_adj(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(float(a) - float(b)) <= tol
