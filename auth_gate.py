"""
auth_gate.py — Google-only sign-in gate for the public platform.

Uses Streamlit's native OIDC auth (st.login / st.user, Streamlit >= 1.42) with
Google as the sole provider. Configuration lives in secrets ([auth] block —
see SETUP_PUBLIC.md). When no [auth] secrets exist (local dev), the gate
degrades to DEV MODE: no login screen, the admin email is assumed.
"""
import streamlit as st

DEV_EMAIL = "ninadshraddhag@gmail.com"


def auth_configured() -> bool:
    try:
        return "auth" in st.secrets and bool(st.secrets["auth"].get("client_id"))
    except Exception:
        return False


def require_login(brand_html: str | None = None) -> tuple[str, bool]:
    """
    Gate the whole app behind Google sign-in.
    Returns (email, dev_mode). Renders the landing page and halts when the
    visitor is not signed in.
    """
    if not auth_configured():
        st.session_state["user_email"] = DEV_EMAIL
        return DEV_EMAIL, True

    if not getattr(st.user, "is_logged_in", False):
        if brand_html:
            st.markdown(brand_html, unsafe_allow_html=True)
        st.markdown("### Probability-driven intraday backtesting")
        st.markdown(
            "10 years of minute data · NIFTY 50 · BANK NIFTY · NQ · XAUUSD  \n"
            "Explore ORB/IB probabilities, backtest day-wise strategies, and "
            "browse every historical setup — free to sign in, backtest runs "
            "are metered by credits.")
        st.divider()
        c = st.columns([1, 2])
        with c[0]:
            if st.button("🔐 Sign in with Google", type="primary",
                         use_container_width=True):
                st.login()
        st.caption("We only receive your Google email address — no password, "
                   "no other data. Educational & research tool; not investment "
                   "advice.")
        st.stop()

    email = (getattr(st.user, "email", "") or "").strip().lower()
    if not email:
        st.error("Google sign-in did not return an email address. Please sign "
                 "out and try again.")
        if st.button("Sign out"):
            st.logout()
        st.stop()
    st.session_state["user_email"] = email
    return email, False


def logout_button():
    if auth_configured():
        if st.button("🚪 Sign out", use_container_width=True):
            st.logout()
