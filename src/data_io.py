from pathlib import Path
import re

import pandas as pd
import streamlit as st


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


def apply_strategy_overrides(strategy_df: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    out = strategy_df.copy()
    if out.empty:
        return out
    out["State"] = out["State"].astype(str).str.upper()
    if not overrides:
        return out
    ov = {
        str(k or "").strip().upper(): str(v or "").strip()
        for k, v in overrides.items()
        if str(k or "").strip() and str(v or "").strip()
    }
    if not ov:
        return out
    out["Strategy Bucket"] = out["State"].map(ov).fillna(out["Strategy Bucket"])
    missing_states = [s for s in ov.keys() if s not in set(out["State"].tolist())]
    if missing_states:
        add_df = pd.DataFrame(
            {"State": missing_states, "Strategy Bucket": [ov[s] for s in missing_states]}
        )
        out = pd.concat([out, add_df], ignore_index=True)
    return out.drop_duplicates("State")


def file_mtime(path: str) -> float:
    try:
        return float(Path(path).stat().st_mtime)
    except Exception:
        return 0.0
