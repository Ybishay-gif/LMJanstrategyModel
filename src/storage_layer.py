import base64
import json
import os
from pathlib import Path
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import OVERRIDES_PATH, AUTH_USERS_PATH

ANALYTICS_PRESETS_PATH = Path("data/analytics_presets.json")
STATE_STRATEGY_OVERRIDES_PATH = Path("data/state_strategy_overrides.json")
STRATEGY_PROFILES_PATH = Path("data/strategy_profiles.json")
AUTH_USERS_STORAGE_PATH = AUTH_USERS_PATH


def _get_secret(name: str) -> str:
    val = os.getenv(name, "")
    if val:
        return str(val).strip()
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return ""


def _github_cfg() -> dict:
    token = _get_secret("PERSIST_GITHUB_TOKEN") or _get_secret("GITHUB_PERSIST_TOKEN")
    repo = _get_secret("PERSIST_GITHUB_REPO") or _get_secret("GITHUB_PERSIST_REPO")
    branch = _get_secret("PERSIST_GITHUB_BRANCH") or "main"
    base_path = _get_secret("PERSIST_GITHUB_BASE_PATH") or ""
    return {
        "enabled": bool(token and repo),
        "token": token,
        "repo": repo,
        "branch": branch,
        "base_path": base_path.strip("/"),
    }


def persistence_backend_name() -> str:
    cfg = _github_cfg()
    return "github" if cfg["enabled"] else "local"


def _join_remote_path(local_path: Path) -> str:
    cfg = _github_cfg()
    rp = str(local_path).replace("\\", "/").lstrip("./")
    if cfg["base_path"]:
        return f"{cfg['base_path']}/{rp}".strip("/")
    return rp


def _github_get_json(local_path: Path) -> tuple[Optional[dict], Optional[str], str]:
    cfg = _github_cfg()
    if not cfg["enabled"]:
        return None, None, "GitHub persistence not configured."
    remote_path = _join_remote_path(local_path)
    url = f"https://api.github.com/repos/{cfg['repo']}/contents/{remote_path}?ref={cfg['branch']}"
    req = Request(url, headers={"Authorization": f"Bearer {cfg['token']}", "Accept": "application/vnd.github+json"})
    try:
        with urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        raw = payload.get("content", "")
        if not raw:
            return {}, payload.get("sha"), ""
        decoded = base64.b64decode(raw.encode("utf-8")).decode("utf-8")
        data = json.loads(decoded)
        if not isinstance(data, dict):
            return {}, payload.get("sha"), ""
        return data, payload.get("sha"), ""
    except HTTPError as e:
        if e.code == 404:
            return {}, None, ""
        return None, None, f"GitHub read failed ({e.code})."
    except URLError:
        return None, None, "GitHub read failed (network)."
    except Exception:
        return None, None, "GitHub read failed."


def _github_put_json(local_path: Path, data: dict, message: str) -> tuple[bool, str]:
    cfg = _github_cfg()
    if not cfg["enabled"]:
        return False, "GitHub persistence not configured."
    remote_path = _join_remote_path(local_path)
    current, sha, err = _github_get_json(local_path)
    if current is None and err:
        return False, err
    content = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    body = {
        "message": message,
        "content": base64.b64encode(content).decode("utf-8"),
        "branch": cfg["branch"],
    }
    if sha:
        body["sha"] = sha
    url = f"https://api.github.com/repos/{cfg['repo']}/contents/{remote_path}"
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="PUT",
        headers={"Authorization": f"Bearer {cfg['token']}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urlopen(req, timeout=15):
            return True, ""
    except HTTPError as e:
        return False, f"GitHub write failed ({e.code})."
    except URLError:
        return False, "GitHub write failed (network)."
    except Exception:
        return False, "GitHub write failed."


def _read_local_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_local_json(path: Path, data: dict) -> tuple[bool, str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        return True, ""
    except Exception:
        return False, "Failed to write local storage file."


def _clean_state_strategy_overrides(data: dict) -> dict:
    out = {}
    for k, v in (data or {}).items():
        ks = str(k or "").strip().upper()
        vs = str(v or "").strip()
        if len(ks) == 2 and vs:
            out[ks] = vs
    return out


def _clean_strategy_profiles(data: dict) -> dict:
    out = {}
    for k, v in (data or {}).items():
        sk = str(k or "").strip()
        if not sk or not isinstance(v, dict):
            continue
        out[sk] = v
    return out


def _clean_any(data: dict) -> dict:
    return data if isinstance(data, dict) else {}


def _load_json(path: Path, cleaner: Callable[[dict], dict]) -> dict:
    cfg = _github_cfg()
    if cfg["enabled"]:
        remote_data, _, err = _github_get_json(path)
        if remote_data is not None:
            cleaned = cleaner(remote_data)
            # Keep local copy in sync for fast fallback.
            _write_local_json(path, cleaned)
            return cleaned
        # On remote failure, use local fallback.
        return cleaner(_read_local_json(path))
    return cleaner(_read_local_json(path))


def _save_json(path: Path, payload: dict, cleaner: Callable[[dict], dict], commit_msg: str, fail_msg: str) -> tuple[bool, str]:
    cleaned = cleaner(payload)
    ok_local, err_local = _write_local_json(path, cleaned)
    if not ok_local:
        return False, err_local or fail_msg

    cfg = _github_cfg()
    if cfg["enabled"]:
        ok_remote, err_remote = _github_put_json(path, cleaned, commit_msg)
        if not ok_remote:
            return False, err_remote or fail_msg
    return True, ""


def load_overrides_from_disk() -> dict:
    return _load_json(OVERRIDES_PATH, _clean_any)


def save_overrides_to_disk(overrides: dict) -> tuple[bool, str]:
    return _save_json(
        OVERRIDES_PATH,
        overrides,
        _clean_any,
        "Update manual bid overrides",
        "Failed to write overrides file.",
    )


def load_analytics_presets() -> dict:
    return _load_json(ANALYTICS_PRESETS_PATH, _clean_any)


def save_analytics_presets(presets: dict) -> tuple[bool, str]:
    return _save_json(
        ANALYTICS_PRESETS_PATH,
        presets,
        _clean_any,
        "Update analytics presets",
        "Failed to write analytics presets file.",
    )


def load_state_strategy_overrides() -> dict:
    return _load_json(STATE_STRATEGY_OVERRIDES_PATH, _clean_state_strategy_overrides)


def save_state_strategy_overrides(overrides: dict) -> tuple[bool, str]:
    return _save_json(
        STATE_STRATEGY_OVERRIDES_PATH,
        overrides,
        _clean_state_strategy_overrides,
        "Update state strategy overrides",
        "Failed to write state strategy overrides.",
    )


def load_strategy_profiles() -> dict:
    return _load_json(STRATEGY_PROFILES_PATH, _clean_strategy_profiles)


def save_strategy_profiles(profiles: dict) -> tuple[bool, str]:
    return _save_json(
        STRATEGY_PROFILES_PATH,
        profiles,
        _clean_strategy_profiles,
        "Update strategy profiles",
        "Failed to write strategy profiles.",
    )


def load_auth_users_store() -> dict:
    return _load_json(AUTH_USERS_STORAGE_PATH, _clean_any)


def save_auth_users_store(users: dict) -> tuple[bool, str]:
    return _save_json(
        AUTH_USERS_STORAGE_PATH,
        users,
        _clean_any,
        "Update auth users",
        "Failed to save user credentials.",
    )
