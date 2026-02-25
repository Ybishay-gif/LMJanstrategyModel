import json
from pathlib import Path

from config import OVERRIDES_PATH

ANALYTICS_PRESETS_PATH = Path("data/analytics_presets.json")


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
