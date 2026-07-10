"""
onboarding.py — first-login welcome tour, per-mode "How to use" guides and the
Guide & Glossary page. Content lives in guide_content.py (pure data); this
module degrades gracefully to minimal built-in text if that file is absent.
"""
import streamlit as st

import credits

try:
    import guide_content as GC
except Exception:
    GC = None

_FALLBACK_TOUR = [
    {"title": "Welcome to Sharp",
     "body": "Probability-driven intraday backtesting on 10 years of minute "
             "data — NIFTY 50, BANK NIFTY, NQ and XAUUSD. Explore the odds, "
             "then backtest them."},
    {"title": "Credits",
     "body": "Browsing probabilities is **free**. Each backtest/optimizer run "
             "costs **1 credit**. You start with a free grant; contact the "
             "admin for more."},
    {"title": "Disclaimer",
     "body": "This is an **educational and research tool only** — not "
             "investment advice. Past performance does not guarantee future "
             "results. Trade at your own risk."},
]


def _tour_pages():
    return (GC.WELCOME_TOUR if GC and getattr(GC, "WELCOME_TOUR", None)
            else _FALLBACK_TOUR)


def maybe_welcome(email: str, user: dict):
    """Show the welcome tour + disclaimer once per user (tracked in the DB)."""
    if user.get("seen_tour") or st.session_state.get("_tour_done"):
        return

    @st.dialog("👋 Welcome — 60-second tour", width="large")
    def _dlg():
        for p in _tour_pages():
            st.markdown(f"##### {p['title']}")
            st.markdown(p["body"])
            st.divider()
        if st.button("✅ I understand & accept — let's go", type="primary",
                     use_container_width=True):
            st.session_state["_tour_done"] = True
            try:
                credits.set_flag(email, "seen_tour", True)
                credits.set_flag(email, "accepted_terms", "now()")
            except Exception:
                pass
            st.rerun()

    _dlg()


def mode_guide(mode_name: str):
    """'How to use this tab' expander — drop at the top of any mode."""
    g = (GC.MODE_GUIDES.get(mode_name)
         if GC and getattr(GC, "MODE_GUIDES", None) else None)
    if not g:
        return
    with st.expander(f"❓ How to use — {mode_name}", expanded=False):
        st.caption(g.get("tagline", ""))
        for i, s in enumerate(g.get("steps", []), 1):
            st.markdown(f"**{i}.** {s}")
        tips = g.get("tips", [])
        if tips:
            st.markdown("**Tips**")
            for t in tips:
                st.markdown(f"- {t}")
        ex = g.get("example")
        if ex:
            st.info(ex)


def guide_page():
    """📖 Guide & Glossary mode — full walkthroughs + every term explained."""
    st.title("📖 Guide & Glossary")
    if GC is None:
        st.info("The full guide is being prepared — check back soon.")
        for p in _FALLBACK_TOUR:
            st.markdown(f"#### {p['title']}")
            st.markdown(p["body"])
        return

    tabs = st.tabs(["🚀 Getting started", "🧭 Mode walkthroughs", "📚 Glossary"])
    with tabs[0]:
        for p in _tour_pages():
            st.markdown(f"#### {p['title']}")
            st.markdown(p["body"])
    with tabs[1]:
        for mode, g in GC.MODE_GUIDES.items():
            with st.expander(f"**{mode}** — {g.get('tagline','')}"):
                for i, s in enumerate(g.get("steps", []), 1):
                    st.markdown(f"**{i}.** {s}")
                for t in g.get("tips", []):
                    st.markdown(f"- 💡 {t}")
                if g.get("example"):
                    st.info(g["example"])
    with tabs[2]:
        q = st.text_input("Search terms", key="gloss_q",
                          placeholder="e.g. extension, PDH, R-multiple")
        terms = GC.GLOSSARY
        if q.strip():
            ql = q.lower()
            terms = {k: v for k, v in terms.items()
                     if ql in k.lower() or ql in v.lower()}
        for term in sorted(terms):
            st.markdown(f"**{term}** — {terms[term]}")
        if not terms:
            st.caption("No matching terms.")
