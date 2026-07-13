"""
credits.py — per-user credit ledger for the public platform.

Backend: Supabase Postgres via PostgREST (secrets [supabase] url + service_key).
Fallback: local JSON file (analysis/credits_local.json, gitignored) so the app
keeps working in local dev with no secrets configured.

Concurrency: charging goes through the `charge_credits` SQL function (atomic
UPDATE ... WHERE credits >= n), so two simultaneous clicks can't double-spend.
See SETUP_PUBLIC.md for the paste-ready SQL.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

import requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_STORE = os.path.join(HERE, "analysis", "credits_local.json")
_LOCK = threading.Lock()

ADMIN_EMAILS = {"ninadshraddhag@gmail.com"}
SUPPORT_EMAIL = "sharpbacktester@gmail.com"   # public help/credits contact
STARTER_CREDITS = 20          # granted automatically on first login
ACTION_COSTS = {              # credits per metered action (default 1)
    "daywise_run": 1,
    "daywise_optimizer": 1,
    "daywise_combined_optimizer": 1,
    "edge_optimizer": 1,
    "ib50_optimizer": 1,
    "advanced_run": 1,
    "advanced_optimizer": 1,
}


# ─── backend selection ────────────────────────────────────────────────────────

def _sb():
    """(url, service_key) when Supabase is configured, else None."""
    try:
        s = st.secrets["supabase"]
        url, key = s["url"].rstrip("/"), s["service_key"]
        if url and key:
            return url, key
    except Exception:
        pass
    return None


def backend_name() -> str:
    return "supabase" if _sb() else "local file (dev)"


def _hdrs(key):
    return {"apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json", "Prefer": "return=representation"}


# ─── local JSON fallback ──────────────────────────────────────────────────────

def _local_load():
    if os.path.exists(LOCAL_STORE):
        try:
            return json.load(open(LOCAL_STORE, encoding="utf-8"))
        except Exception:
            pass
    return {"users": {}, "log": []}


def _local_save(d):
    os.makedirs(os.path.dirname(LOCAL_STORE), exist_ok=True)
    json.dump(d, open(LOCAL_STORE, "w", encoding="utf-8"), indent=1, default=str)


def _now():
    return datetime.now(timezone.utc).isoformat()


# ─── public API ───────────────────────────────────────────────────────────────

def ensure_user(email: str) -> dict:
    """Fetch the user row, auto-provisioning first-time visitors."""
    email = email.strip().lower()
    sb = _sb()
    if sb:
        url, key = sb
        r = requests.get(f"{url}/rest/v1/users", headers=_hdrs(key),
                         params={"email": f"eq.{email}", "select": "*"}, timeout=10)
        rows = r.json() if r.ok else []
        if rows:
            return rows[0]
        row = {"email": email, "credits": STARTER_CREDITS,
               "is_admin": email in ADMIN_EMAILS}
        r = requests.post(f"{url}/rest/v1/users", headers=_hdrs(key),
                          json=row, timeout=10)
        if r.ok and r.json():
            return r.json()[0]
        # lost a provisioning race — re-read
        r = requests.get(f"{url}/rest/v1/users", headers=_hdrs(key),
                         params={"email": f"eq.{email}", "select": "*"}, timeout=10)
        rows = r.json() if r.ok else []
        return rows[0] if rows else {"email": email, "credits": 0}
    with _LOCK:
        d = _local_load()
        u = d["users"].get(email)
        if not u:
            u = {"email": email, "credits": STARTER_CREDITS, "total_used": 0,
                 "is_admin": email in ADMIN_EMAILS, "seen_tour": False,
                 "accepted_terms": None, "first_seen": _now(), "last_seen": _now()}
            d["users"][email] = u
            _local_save(d)
        return dict(u)


def balance(email: str) -> int:
    u = ensure_user(email)
    return int(u.get("credits", 0))


def is_admin(email: str) -> bool:
    """
    Admin = flagged email AND the admin PIN entered this session. With the
    honor-system sign-up form, merely typing the admin's email must not grant
    the panel or unlimited runs.
    """
    flagged = (email.strip().lower() in ADMIN_EMAILS
               or bool(ensure_user(email).get("is_admin")))
    if not flagged:
        return False
    try:
        return bool(st.session_state.get("admin_ok"))
    except Exception:
        return True          # non-Streamlit contexts (scripts, tests)


def signup(email: str, name: str) -> dict:
    """First-touch registration from the sign-up form: provision the account,
    store the name (best-effort), and log the signup so the admin sees every
    new user in the panel."""
    email = email.strip().lower()
    existed = False
    sb = _sb()
    if sb:
        url, key = sb
        r = requests.get(f"{url}/rest/v1/users", headers=_hdrs(key),
                         params={"email": f"eq.{email}", "select": "email"},
                         timeout=10)
        existed = bool(r.ok and r.json())
    else:
        existed = email in _local_load()["users"]
    user = ensure_user(email)
    try:
        set_flag(email, "name", name)      # ignored if the column doesn't exist
    except Exception:
        pass
    if not existed:
        if sb:
            url, key = sb
            try:
                requests.post(f"{url}/rest/v1/usage_log", headers=_hdrs(key),
                              json={"email": email, "action": f"signup:{name}",
                                    "credits": 0}, timeout=10)
            except Exception:
                pass
        else:
            with _LOCK:
                d = _local_load()
                d["log"].append({"email": email, "action": f"signup:{name}",
                                 "credits": 0, "ts": _now()})
                _local_save(d)
    return user


def charge(email: str, action: str, n: int | None = None) -> tuple[bool, int]:
    """Atomically deduct credits. Returns (ok, new_balance). Admins ride free."""
    email = email.strip().lower()
    n = ACTION_COSTS.get(action, 1) if n is None else n
    if is_admin(email):
        return True, balance(email)
    sb = _sb()
    if sb:
        url, key = sb
        r = requests.post(f"{url}/rest/v1/rpc/charge_credits", headers=_hdrs(key),
                          json={"p_email": email, "p_n": n, "p_action": action},
                          timeout=10)
        if not r.ok:
            return False, -1
        nb = int(r.json())
        return (nb >= 0), max(nb, 0)
    with _LOCK:
        d = _local_load()
        u = d["users"].setdefault(email, {"email": email, "credits": 0,
                                          "total_used": 0})
        if u.get("credits", 0) < n:
            return False, int(u.get("credits", 0))
        u["credits"] -= n
        u["total_used"] = u.get("total_used", 0) + n
        u["last_seen"] = _now()
        d["log"].append({"email": email, "action": action, "credits": n, "ts": _now()})
        _local_save(d)
        return True, int(u["credits"])


def add_credits(email: str, n: int) -> int:
    """Admin grant (n may be negative to revoke). Returns new balance."""
    email = email.strip().lower()
    sb = _sb()
    if sb:
        url, key = sb
        r = requests.post(f"{url}/rest/v1/rpc/add_credits", headers=_hdrs(key),
                          json={"p_email": email, "p_n": int(n)}, timeout=10)
        return int(r.json()) if r.ok else -1
    with _LOCK:
        d = _local_load()
        u = d["users"].setdefault(email, {"email": email, "credits": 0, "total_used": 0})
        u["credits"] = int(u.get("credits", 0)) + int(n)
        d["log"].append({"email": email, "action": "admin_grant",
                         "credits": -int(n), "ts": _now()})
        _local_save(d)
        return int(u["credits"])


def set_flag(email: str, field: str, value) -> None:
    """Set a user column (seen_tour / accepted_terms / is_admin)."""
    email = email.strip().lower()
    sb = _sb()
    if sb:
        url, key = sb
        requests.patch(f"{url}/rest/v1/users", headers=_hdrs(key),
                       params={"email": f"eq.{email}"},
                       json={field: value}, timeout=10)
        return
    with _LOCK:
        d = _local_load()
        if email in d["users"]:
            d["users"][email][field] = value
            _local_save(d)


def list_users() -> list[dict]:
    sb = _sb()
    if sb:
        url, key = sb
        r = requests.get(f"{url}/rest/v1/users", headers=_hdrs(key),
                         params={"select": "*", "order": "last_seen.desc",
                                 "limit": "1000"}, timeout=15)
        return r.json() if r.ok else []
    return list(_local_load()["users"].values())


def usage_log(limit: int = 500) -> list[dict]:
    sb = _sb()
    if sb:
        url, key = sb
        r = requests.get(f"{url}/rest/v1/usage_log", headers=_hdrs(key),
                         params={"select": "*", "order": "ts.desc",
                                 "limit": str(limit)}, timeout=15)
        return r.json() if r.ok else []
    return list(reversed(_local_load()["log"][-limit:]))


# ─── streamlit UI helper ──────────────────────────────────────────────────────

def try_charge(action: str) -> bool:
    """Charge the signed-in user for a metered run; show UI feedback.
    Returns True when the run may proceed."""
    email = st.session_state.get("user_email")
    if not email:
        st.error("Sign-in required.")
        return False
    n = ACTION_COSTS.get(action, 1)
    ok, bal = charge(email, action, n)
    if not ok:
        st.error(f"⛔ You're out of credits (this run costs {n}). "
                 f"Email **{SUPPORT_EMAIL}** to top up your balance.")
        return False
    st.session_state["credit_balance"] = bal
    if not is_admin(email):
        st.toast(f"−{n} credit · {bal} remaining", icon="💳")
    return True
