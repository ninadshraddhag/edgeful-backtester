"""
auth_gate.py — simple sign-up gate: name + email. No passwords, no OAuth.

The earlier Google-OIDC (st.login) flow proved fragile on Streamlit Cloud and
was removed. Visitors sign up / return with just their name + email; accounts
and credits stay keyed by email in credits.py, so nothing else changed.

Admin access is protected by a PIN (st.secrets["app"]["admin_pin"]) — with an
honor-system form, typing the admin's email must NOT grant the admin panel.

DEV MODE: running locally (not on Streamlit Cloud) skips the gate entirely
and assumes the admin identity, exactly as before.
"""
import os
import re
import sys

import streamlit as st

import credits

DEV_EMAIL = "ninadshraddhag@gmail.com"

# Streamlit Community Cloud runs the repo from /mount/src on Linux; the owner's
# dev machine is Windows.
IS_CLOUD = os.path.exists("/mount/src") or (not sys.platform.startswith("win"))

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


def _admin_pin():
    try:
        return str(st.secrets["app"]["admin_pin"]) or None
    except Exception:
        return None


def require_login(brand_html: str | None = None) -> tuple[str, bool]:
    """
    Gate the app behind the sign-up form. Returns (email, dev_mode).
    Renders the sign-up page and halts when the visitor hasn't signed up yet.
    """
    if not IS_CLOUD:                                   # local dev — no gate
        st.session_state["user_email"] = DEV_EMAIL
        st.session_state.setdefault("user_name", "Ninad (dev)")
        st.session_state["admin_ok"] = True
        return DEV_EMAIL, True

    if st.session_state.get("user_email"):
        return st.session_state["user_email"], False

    # ── sign-up / sign-in page ────────────────────────────────────────────────
    if brand_html:
        st.markdown(brand_html, unsafe_allow_html=True)
    st.markdown("### Probability-driven intraday backtesting")
    st.markdown(
        "10 years of minute data · NIFTY 50 · BANK NIFTY · NQ · XAUUSD  \n"
        "Explore ORB/IB probabilities, backtest day-wise strategies, and browse "
        "every historical setup. **Free to join** — backtest runs are metered "
        "by credits (20 free on sign-up).")
    st.divider()

    with st.form("sharp_signup"):
        c = st.columns(2)
        name = c[0].text_input("Your name", placeholder="e.g. Rahul Sharma")
        email = c[1].text_input("Email ID", placeholder="you@example.com")
        code = st.text_input("Access code (optional — for admins only)",
                             type="password",
                             help="Regular users leave this blank.")
        ok = st.form_submit_button("🚀 Start backtesting", type="primary",
                                   use_container_width=True)
    st.caption("By continuing you accept: this is an educational & research "
               "tool only — not investment advice. Past performance does not "
               "guarantee future results.  ·  Help: sharpbacktester@gmail.com")

    if ok:
        email_c = email.strip().lower()
        if not name.strip():
            st.error("Please enter your name.")
        elif not _EMAIL_RE.match(email_c):
            st.error("Please enter a valid email ID.")
        else:
            st.session_state["user_email"] = email_c
            st.session_state["user_name"] = name.strip()
            pin = _admin_pin()
            st.session_state["admin_ok"] = bool(pin) and code.strip() == pin
            credits.signup(email_c, name.strip())
            st.rerun()
    st.stop()


def logout_button():
    if st.button("🚪 Sign out", use_container_width=True):
        for k in ("user_email", "user_name", "admin_ok", "credit_balance",
                  "_tour_done"):
            st.session_state.pop(k, None)
        st.rerun()
