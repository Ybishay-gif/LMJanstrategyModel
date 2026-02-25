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


def file_mtime(path: str) -> float:
    try:
        return float(Path(path).stat().st_mtime)
    except Exception:
        return 0.0
