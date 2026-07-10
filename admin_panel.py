"""
admin_panel.py — 👑 Admin mode, visible only to admin emails.

Users table with balances, credit grant/revoke, usage log, CSV exports.
Backend-agnostic: everything goes through credits.py (Supabase or local dev).
"""
import pandas as pd
import streamlit as st

import credits


def render(admin_email: str):
    st.title("👑 Admin — users & credits")
    st.caption(f"Signed in as **{admin_email}** · ledger backend: "
               f"**{credits.backend_name()}**")

    users = credits.list_users()
    if not users:
        st.info("No users yet. Rows appear here the first time someone signs "
                "in with Google.")
        return

    df = pd.DataFrame(users)
    cols = [c for c in ("email", "credits", "total_used", "is_admin",
                        "accepted_terms", "first_seen", "last_seen") if c in df.columns]
    df = df[cols].sort_values("last_seen" if "last_seen" in cols else "email",
                              ascending=False)

    m = st.columns(4)
    m[0].metric("Users", f"{len(df):,}")
    m[1].metric("Credits outstanding", f"{int(df['credits'].sum()):,}")
    if "total_used" in df.columns:
        m[2].metric("Runs consumed", f"{int(df['total_used'].sum()):,}")
    if "first_seen" in df.columns:
        recent = pd.to_datetime(df["first_seen"], errors="coerce", utc=True)
        m[3].metric("New this week",
                    f"{int((recent > pd.Timestamp.utcnow() - pd.Timedelta(days=7)).sum()):,}")

    st.markdown("#### Users")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇ Export users (CSV)", df.to_csv(index=False).encode(),
                       "users.csv", "text/csv")

    # ── grant / revoke ────────────────────────────────────────────────────────
    st.markdown("#### Grant / revoke credits")
    g = st.columns([3, 1, 1, 1])
    target = g[0].selectbox("User", df["email"].tolist(), key="adm_target")
    amount = g[1].number_input("Credits", 1, 100000, 100, 10, key="adm_amount")
    if g[2].button("➕ Grant", use_container_width=True, key="adm_grant"):
        nb = credits.add_credits(target, int(amount))
        st.success(f"{target} → new balance **{nb}**")
        st.rerun()
    if g[3].button("➖ Revoke", use_container_width=True, key="adm_revoke"):
        nb = credits.add_credits(target, -int(amount))
        st.warning(f"{target} → new balance **{nb}**")
        st.rerun()

    # ── usage log ─────────────────────────────────────────────────────────────
    st.markdown("#### Usage log (latest 500)")
    log = credits.usage_log(500)
    if log:
        lf = pd.DataFrame(log)
        st.dataframe(lf, use_container_width=True, hide_index=True, height=320)
        st.download_button("⬇ Export log (CSV)", lf.to_csv(index=False).encode(),
                           "usage_log.csv", "text/csv")
        if "action" in lf.columns:
            st.markdown("**Runs by action**")
            st.dataframe(lf[lf["action"] != "admin_grant"]["action"]
                         .value_counts().rename_axis("action").reset_index(name="runs"),
                         hide_index=True)
    else:
        st.info("No usage yet.")
