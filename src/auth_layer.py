import os
import json
import hmac
import base64
import hashlib
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from urllib.parse import urlencode
from typing import Optional

import pandas as pd
import streamlit as st

from config import AUTH_USERS_PATH, AUTH_ALLOWLIST_PATH, ADMIN_EMAIL, SESSION_TTL_DAYS
from ui_utils import render_formatted_table

def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def load_allowed_emails() -> set[str]:
    out: set[str] = set()
    raw_env = os.getenv("APP_ALLOWED_EMAILS", "")
    if raw_env:
        out.update({normalize_email(x) for x in raw_env.split(",") if normalize_email(x)})
    try:
        if AUTH_ALLOWLIST_PATH.exists():
            for ln in AUTH_ALLOWLIST_PATH.read_text(errors="ignore").splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                em = normalize_email(ln)
                if em:
                    out.add(em)
    except Exception:
        pass
    return out


def save_allowed_emails(emails: set[str]) -> tuple[bool, str]:
    try:
        AUTH_ALLOWLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted({normalize_email(e) for e in emails if normalize_email(e)})
        AUTH_ALLOWLIST_PATH.write_text("\n".join(rows) + ("\n" if rows else ""))
        return True, ""
    except Exception:
        return False, "Failed to save allowlist."


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_auth_users() -> dict:
    try:
        if not AUTH_USERS_PATH.exists():
            return {}
        data = json.loads(AUTH_USERS_PATH.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_auth_users(users: dict) -> tuple[bool, str]:
    try:
        AUTH_USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        AUTH_USERS_PATH.write_text(json.dumps(users, indent=2))
        return True, ""
    except Exception as exc:
        return False, f"Failed to save user credentials: {exc}"


def hash_password(password: str) -> tuple[str, str, int]:
    iterations = 240_000
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return (
        base64.b64encode(salt).decode("utf-8"),
        base64.b64encode(digest).decode("utf-8"),
        iterations,
    )


def verify_password(password: str, salt_b64: str, hash_b64: str, iterations: int) -> bool:
    try:
        salt = base64.b64decode(salt_b64.encode("utf-8"))
        expected = base64.b64decode(hash_b64.encode("utf-8"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def make_invite_token() -> str:
    return secrets.token_urlsafe(24)


def make_session_token() -> str:
    return secrets.token_urlsafe(32)


def _session_signing_key() -> bytes:
    raw = (
        os.getenv("APP_SESSION_SIGNING_KEY", "").strip()
        or os.getenv("APP_BASE_URL", "").strip()
        or "insurance-growth-navigator-session-key"
    )
    return raw.encode("utf-8")


def _request_fingerprint(headers: Optional[dict] = None) -> str:
    """
    Build a lightweight client fingerprint to reduce session-token sharing risk.
    Not a strong security boundary, but enough to block copied URL sessions across users.
    """
    try:
        h = headers if isinstance(headers, dict) else dict(getattr(st, "context").headers or {})
    except Exception:
        h = {}
    ua = str(h.get("user-agent", "")).strip().lower()
    al = str(h.get("accept-language", "")).strip().lower()
    xff = str(h.get("x-forwarded-for", "")).split(",")[0].strip().lower()
    raw = f"{ua}|{al}|{xff}"
    if not raw.replace("|", ""):
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_stateless_session_token(email: str, fingerprint: str = "") -> str:
    exp = int((datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)).timestamp())
    payload = {"e": normalize_email(email), "x": exp}
    fp = str(fingerprint or "").strip()
    if fp:
        payload["f"] = fp
    payload_raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_raw).decode("utf-8").rstrip("=")
    sig = hmac.new(_session_signing_key(), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    return f"v2.{payload_b64}.{sig_b64}"


def resolve_stateless_session_token(token: str, current_fingerprint: str = "") -> Optional[str]:
    t = str(token or "").strip()
    if not t.startswith("v2."):
        return None
    try:
        _, payload_b64, sig_b64 = t.split(".", 2)
        expected_sig = hmac.new(_session_signing_key(), payload_b64.encode("utf-8"), hashlib.sha256).digest()
        actual_sig = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        payload_raw = base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4))
        payload = json.loads(payload_raw.decode("utf-8"))
        email = normalize_email(payload.get("e", ""))
        exp = int(payload.get("x", 0))
        token_fp = str(payload.get("f", "")).strip()
        if not email or "@" not in email:
            return None
        if exp <= int(datetime.now(timezone.utc).timestamp()):
            return None
        if token_fp and token_fp != str(current_fingerprint or "").strip():
            return None
        allowed = load_allowed_emails()
        if email not in allowed:
            return None
        return email
    except Exception:
        return None


def build_invite_link(token: str) -> str:
    base_url = str(os.getenv("APP_BASE_URL", "")).strip().rstrip("/")
    if not base_url:
        return f"?invite_token={token}"
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}invite_token={token}"


def issue_session(users: dict, email: str) -> tuple[dict, str]:
    e = normalize_email(email)
    rec = users.get(e, {}) if isinstance(users.get(e), dict) else {}
    token = make_session_token()
    rec["session_token"] = token
    rec["session_expires_at"] = (datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)).isoformat()
    users[e] = rec
    return users, token


def resolve_session_token(users: dict, token: str) -> Optional[str]:
    t = str(token or "").strip()
    if not t:
        return None
    stateless = resolve_stateless_session_token(t, _request_fingerprint())
    if stateless:
        return stateless
    now = datetime.now(timezone.utc)
    for email, rec in users.items():
        if not isinstance(rec, dict):
            continue
        if str(rec.get("session_token", "")) != t:
            continue
        if not bool(rec.get("active", True)):
            return None
        try:
            exp = datetime.fromisoformat(str(rec.get("session_expires_at", "")))
        except Exception:
            return None
        if exp < now:
            return None
        return normalize_email(email)
    return None


def send_invite_email(email: str, link: str) -> tuple[bool, str]:
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587") or "587")
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASS", "").strip()
    sender = os.getenv("SMTP_FROM", user).strip()
    use_tls = str(os.getenv("SMTP_USE_TLS", "true")).strip().lower() in {"1", "true", "yes", "y"}
    if not host or not sender:
        return False, "SMTP is not configured (`SMTP_HOST` / `SMTP_FROM`)."
    try:
        msg = EmailMessage()
        msg["Subject"] = "Invite: Insurance Growth Navigator"
        msg["From"] = sender
        msg["To"] = email
        msg.set_content(
            "You were invited to Insurance Growth Navigator.\n"
            f"Use this link to create your password:\n{link}\n\n"
            "If you did not expect this invite, ignore this email."
        )
        with smtplib.SMTP(host, port, timeout=20) as s:
            if use_tls:
                s.starttls()
            if user and password:
                s.login(user, password)
            s.send_message(msg)
        return True, ""
    except Exception as exc:
        return False, f"Failed to send invite email: {exc}"


def resolve_invite_token(users: dict, token: str) -> Optional[str]:
    t = str(token or "").strip()
    if not t:
        return None
    now = datetime.now(timezone.utc)
    for email, rec in users.items():
        if not isinstance(rec, dict):
            continue
        if str(rec.get("invite_token", "")) != t:
            continue
        try:
            exp = datetime.fromisoformat(str(rec.get("invite_expires_at", "")))
        except Exception:
            exp = None
        if exp is None or exp < now:
            return None
        if not bool(rec.get("active", True)):
            return None
        return normalize_email(email)
    return None


def render_auth_gate() -> bool:
    if st.session_state.get("auth_ok", False):
        return True

    allowed = load_allowed_emails()
    users = load_auth_users()
    qp_session = st.query_params.get("session_token")
    if isinstance(qp_session, list):
        qp_session = qp_session[0] if qp_session else ""
    remembered_email = resolve_session_token(users, str(qp_session or ""))
    if qp_session and not remembered_email and "session_token" in st.query_params:
        # Drop stale/invalid/shared token so user sees login instead of silent bypass attempts.
        del st.query_params["session_token"]
    if remembered_email:
        st.session_state["auth_ok"] = True
        st.session_state["auth_user"] = remembered_email
        st.session_state["auth_email"] = remembered_email
        st.session_state["auth_stage"] = "password"
        return True

    qp_token = st.query_params.get("invite_token")
    if isinstance(qp_token, list):
        qp_token = qp_token[0] if qp_token else ""
    invited_email = resolve_invite_token(users, str(qp_token or ""))
    if invited_email:
        rec = users.get(invited_email, {}) if isinstance(users.get(invited_email), dict) else {}
        if rec.get("password_hash"):
            st.session_state["auth_email"] = invited_email
            st.session_state["auth_stage"] = "password"
            if "invite_token" in st.query_params:
                del st.query_params["invite_token"]
        else:
            st.session_state["auth_email"] = invited_email
            st.session_state["auth_stage"] = "create"
    elif qp_token:
        # Invalid/expired invite token should not trap user in signup flow.
        if "invite_token" in st.query_params:
            del st.query_params["invite_token"]

    st.title("🔐 Secure Login")
    if not allowed:
        st.error("No allowlist configured. Add emails to `data/allowed_emails.txt`.")
        return False

    stage = st.session_state.get("auth_stage", "email")
    staged_email = normalize_email(st.session_state.get("auth_email", ""))

    if stage == "email":
        with st.form("auth_email_form", clear_on_submit=False):
            email = st.text_input("Work email", value=staged_email, placeholder="name@company.com")
            submitted = st.form_submit_button("Next")
        if submitted:
            e = normalize_email(email)
            if "@" not in e:
                st.error("Enter a valid email.")
                return False
            if e not in allowed:
                st.error("Email is not authorized.")
                return False
            existing = users.get(e, {})
            if isinstance(existing, dict) and not bool(existing.get("active", True)):
                st.error("User access is disabled.")
                return False
            st.session_state["auth_email"] = e
            if isinstance(users.get(e), dict) and users[e].get("password_hash"):
                st.session_state["auth_stage"] = "password"
            else:
                st.session_state["auth_stage"] = "create"
            st.rerun()
        return False

    if stage == "create":
        st.info(f"First-time setup for: `{staged_email}`")
        with st.form("auth_create_form", clear_on_submit=False):
            p1 = st.text_input("Create password", type="password")
            p2 = st.text_input("Confirm password", type="password")
            c1, c2 = st.columns(2)
            create = c1.form_submit_button("Create Password")
            back = c2.form_submit_button("Back")
        if back:
            st.session_state["auth_stage"] = "email"
            st.rerun()
        if create:
            if len(p1) < 8:
                st.error("Password must be at least 8 characters.")
                return False
            if p1 != p2:
                st.error("Passwords do not match.")
                return False
            salt_b64, hash_b64, iters = hash_password(p1)
            prev = users.get(staged_email, {}) if isinstance(users.get(staged_email), dict) else {}
            users[staged_email] = {
                "email": staged_email,
                "salt": salt_b64,
                "password_hash": hash_b64,
                "iterations": int(iters),
                "active": bool(prev.get("active", True)),
                "invited_at": prev.get("invited_at", ""),
                "last_login_at": now_iso(),
                "invite_token": "",
                "invite_expires_at": "",
            }
            ok, err = save_auth_users(users)
            if not ok:
                st.error(err)
                return False
            users, sess = issue_session(users, staged_email)
            ok2, err2 = save_auth_users(users)
            if not ok2:
                st.error(err2)
                return False
            st.query_params["session_token"] = make_stateless_session_token(
                staged_email, _request_fingerprint()
            )
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = staged_email
            st.session_state["auth_stage"] = "password"
            if "invite_token" in st.query_params:
                del st.query_params["invite_token"]
            st.rerun()
        return False

    if stage == "password":
        st.caption(f"Email: `{staged_email}`")
        with st.form("auth_password_form", clear_on_submit=False):
            pw = st.text_input("Password", type="password")
            c1, c2 = st.columns(2)
            login = c1.form_submit_button("Login")
            switch = c2.form_submit_button("Use Another Email")
        if switch:
            st.session_state["auth_stage"] = "email"
            st.session_state["auth_email"] = ""
            st.rerun()
        if login:
            rec = users.get(staged_email, {})
            if not isinstance(rec, dict) or not rec.get("password_hash"):
                st.error("User is not provisioned. Run first-time setup.")
                st.session_state["auth_stage"] = "email"
                return False
            if not bool(rec.get("active", True)):
                st.error("User access is disabled.")
                return False
            ok = verify_password(
                pw,
                str(rec.get("salt", "")),
                str(rec.get("password_hash", "")),
                int(rec.get("iterations", 240000)),
            )
            if not ok:
                st.error("Invalid email or password.")
                return False
            rec["last_login_at"] = now_iso()
            users[staged_email] = rec
            users, sess = issue_session(users, staged_email)
            ok3, err3 = save_auth_users(users)
            if not ok3:
                st.error(err3)
                return False
            st.query_params["session_token"] = make_stateless_session_token(
                staged_email, _request_fingerprint()
            )
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = staged_email
            st.rerun()
        return False

    st.session_state["auth_stage"] = "email"
    return False


def render_settings_panel() -> None:
    current_user = normalize_email(st.session_state.get("auth_user", ""))
    if current_user != ADMIN_EMAIL:
        st.error("Settings is available for admin only.")
        return

    users = load_auth_users()
    allowed = load_allowed_emails()

    st.subheader("⚙️ Access Settings")
    rows = []
    all_emails = sorted(set(allowed) | set(normalize_email(x) for x in users.keys()))
    for e in all_emails:
        rec = users.get(e, {}) if isinstance(users.get(e), dict) else {}
        rows.append(
            {
                "Email": e,
                "Has Password": bool(rec.get("password_hash")),
                "Access": "Active" if bool(rec.get("active", True)) else "Revoked",
                "Last Login": rec.get("last_login_at", ""),
                "Invited At": rec.get("invited_at", ""),
            }
        )
    users_df = pd.DataFrame(rows).sort_values("Email") if rows else pd.DataFrame(columns=["Email", "Has Password", "Access", "Last Login", "Invited At"])
    render_formatted_table(users_df, use_container_width=True)

    st.markdown("**Manage Existing User**")
    if not users_df.empty:
        c1, c2, c3 = st.columns([2, 1, 1])
        picked = c1.selectbox("User", options=users_df["Email"].tolist(), key="settings_user_pick")
        do_revoke = c2.button("Revoke Access", key="settings_revoke")
        do_restore = c3.button("Restore Access", key="settings_restore")
        if do_revoke and picked:
            rec = users.get(picked, {}) if isinstance(users.get(picked), dict) else {}
            rec["email"] = picked
            rec["active"] = False
            users[picked] = rec
            ok, err = save_auth_users(users)
            if ok:
                st.success(f"Revoked access: {picked}")
                st.rerun()
            else:
                st.error(err)
        if do_restore and picked:
            rec = users.get(picked, {}) if isinstance(users.get(picked), dict) else {}
            rec["email"] = picked
            rec["active"] = True
            users[picked] = rec
            allowed.add(picked)
            ok1, err1 = save_auth_users(users)
            ok2, err2 = save_allowed_emails(allowed)
            if ok1 and ok2:
                st.success(f"Restored access: {picked}")
                st.rerun()
            else:
                st.error(err1 or err2)

    st.markdown("**Invite New User**")
    with st.form("invite_form", clear_on_submit=False):
        new_email = st.text_input("Email to invite", placeholder="name@company.com")
        send_btn = st.form_submit_button("Invite")
    if send_btn:
        e = normalize_email(new_email)
        if "@" not in e:
            st.error("Enter a valid email.")
        else:
            allowed.add(e)
            rec = users.get(e, {}) if isinstance(users.get(e), dict) else {}
            token = make_invite_token()
            rec["email"] = e
            rec["active"] = True
            rec["invited_at"] = now_iso()
            rec["invite_token"] = token
            rec["invite_expires_at"] = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
            users[e] = rec
            ok1, err1 = save_allowed_emails(allowed)
            ok2, err2 = save_auth_users(users)
            if not (ok1 and ok2):
                st.error(err1 or err2)
            else:
                link = build_invite_link(token)
                sent, msg = send_invite_email(e, link)
                if sent:
                    st.success(f"Invite sent to {e}.")
                else:
                    st.warning(msg)
                    st.info(f"Share this invite link manually: {link}")
                st.rerun()


def qp_value(name: str, default: str = "") -> str:
    v = st.query_params.get(name, default)
    if isinstance(v, list):
        return str(v[0]) if v else str(default)
    return str(v) if v is not None else str(default)


def build_query_url(updates: dict[str, Optional[str]]) -> str:
    params: dict[str, str] = {}
    for k in st.query_params.keys():
        vv = st.query_params.get(k)
        if isinstance(vv, list):
            vv = vv[0] if vv else ""
        if vv is None:
            continue
        params[str(k)] = str(vv)
    for k, v in updates.items():
        if v is None or str(v) == "":
            params.pop(k, None)
        else:
            params[k] = str(v)
    q = urlencode(params)
    return f"?{q}" if q else "?"


def perform_logout() -> None:
    u = normalize_email(st.session_state.get("auth_user", ""))
    users = load_auth_users()
    rec = users.get(u, {}) if isinstance(users.get(u), dict) else {}
    rec["session_token"] = ""
    rec["session_expires_at"] = ""
    users[u] = rec
    save_auth_users(users)
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = ""
    st.session_state["auth_stage"] = "email"
    st.session_state["auth_email"] = ""
    st.session_state["show_settings"] = False
    for k in ("session_token", "view", "action"):
        if k in st.query_params:
            del st.query_params[k]
    st.rerun()


def render_top_icons(is_admin: bool, settings_view: bool = False) -> None:
    logout_url = build_query_url({"action": "logout", "view": None})
    settings_url = build_query_url({"view": "settings", "action": None}) if is_admin else ""
    back_url = build_query_url({"view": "main", "action": None})
    settings_icon = (
        f"<a href='{settings_url}' title='Settings' style='text-decoration:none;font-size:1.35rem;line-height:1;cursor:pointer;'>⚙️</a>"
        if is_admin
        else "<span title='Settings (admin only)' style='opacity:.35;font-size:1.35rem;line-height:1;cursor:not-allowed;'>⚙️</span>"
    )
    left_icon = (
        f"<a href='{back_url}' title='Back' style='text-decoration:none;font-size:1.35rem;line-height:1;cursor:pointer;'>⬅️</a>"
        if settings_view
        else settings_icon
    )
    st.markdown(
        (
            "<div style='display:flex;justify-content:flex-end;align-items:center;gap:14px;padding-top:4px;'>"
            f"{left_icon}"
            f"<a href='{logout_url}' title='Logout' style='text-decoration:none;font-size:1.35rem;line-height:1;cursor:pointer;'>🚪</a>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
