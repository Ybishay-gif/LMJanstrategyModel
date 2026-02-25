import json
from pathlib import Path

from config import OVERRIDES_PATH

ANALYTICS_PRESETS_PATH = Path("data/analytics_presets.json")
STATE_STRATEGY_OVERRIDES_PATH = Path("data/state_strategy_overrides.json")
STRATEGY_PROFILES_PATH = Path("data/strategy_profiles.json")


def load_overrides_from_disk() -> dict:
    try:
        if not OVERRIDES_PATH.exists():
            return {}
        data = json.loads(OVERRIDES_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_overrides_to_disk(overrides: dict) -> tuple[bool, str]:
    try:
        OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
        return True, ""
    except Exception:
        return False, "Failed to write overrides file."


def load_analytics_presets() -> dict:
    try:
        if not ANALYTICS_PRESETS_PATH.exists():
            return {}
        data = json.loads(ANALYTICS_PRESETS_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_analytics_presets(presets: dict) -> tuple[bool, str]:
    try:
        ANALYTICS_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANALYTICS_PRESETS_PATH.write_text(json.dumps(presets, indent=2))
        return True, ""
    except Exception:
        return False, "Failed to write analytics presets file."


def load_state_strategy_overrides() -> dict:
    try:
        if not STATE_STRATEGY_OVERRIDES_PATH.exists():
            return {}
        data = json.loads(STATE_STRATEGY_OVERRIDES_PATH.read_text())
        if not isinstance(data, dict):
            return {}
        out = {}
        for k, v in data.items():
            ks = str(k or "").strip().upper()
            vs = str(v or "").strip()
            if len(ks) == 2 and vs:
                out[ks] = vs
        return out
    except Exception:
        return {}


def save_state_strategy_overrides(overrides: dict) -> tuple[bool, str]:
    try:
        STATE_STRATEGY_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned = {}
        for k, v in (overrides or {}).items():
            ks = str(k or "").strip().upper()
            vs = str(v or "").strip()
            if len(ks) == 2 and vs:
                cleaned[ks] = vs
        STATE_STRATEGY_OVERRIDES_PATH.write_text(json.dumps(cleaned, indent=2))
        return True, ""
    except Exception:
        return False, "Failed to write state strategy overrides."


def load_strategy_profiles() -> dict:
    try:
        if not STRATEGY_PROFILES_PATH.exists():
            return {}
        data = json.loads(STRATEGY_PROFILES_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_strategy_profiles(profiles: dict) -> tuple[bool, str]:
    try:
        STRATEGY_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned = {}
        for k, v in (profiles or {}).items():
            sk = str(k or "").strip()
            if not sk:
                continue
            if isinstance(v, dict):
                cleaned[sk] = v
        STRATEGY_PROFILES_PATH.write_text(json.dumps(cleaned, indent=2))
        return True, ""
    except Exception:
        return False, "Failed to write strategy profiles."
