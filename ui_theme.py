"""
ui_theme.py — readability CSS the Sharp brand stylesheet (app.py APP_CSS)
doesn't cover: clearly visible scrollbars, readable dropdown menus, and
input-field borders. Base colors live in .streamlit/config.toml.

Usage:  import ui_theme; ui_theme.apply()
Called from app.py inject_style() (all modes) and prob_app._standalone().
"""
import streamlit as st

_CSS = """
<style>
/* ── scrollbars: clearly visible everywhere (tables, sidebar, dropdowns) ── */
*::-webkit-scrollbar        { width: 12px; height: 12px; }
*::-webkit-scrollbar-track  { background: rgba(255,255,255,0.06); border-radius: 8px; }
*::-webkit-scrollbar-thumb  { background: #4a5a75; border-radius: 8px;
                              border: 2px solid transparent; background-clip: padding-box; }
*::-webkit-scrollbar-thumb:hover { background: #67799a; background-clip: padding-box; }
*                           { scrollbar-width: auto; scrollbar-color: #4a5a75 #1c2330; }

/* ── select / multiselect: visible control border + readable menu ── */
div[data-baseweb="select"] > div {
    background-color: #1c2330 !important;
    border-color: #3c4d70 !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] ul {
    background-color: #202a3e !important;
    border: 1px solid #3c4d70 !important;
}
div[data-baseweb="popover"] li:hover { background-color: #2c3a57 !important; }

/* ── text / number / date inputs: visible borders ── */
.stTextInput input, .stNumberInput input, .stDateInput input {
    background-color: #1c2330 !important;
    border: 1px solid #3c4d70 !important;
}

/* ── horizontal dividers: actually visible ── */
hr { border-color: #33415f !important; }
</style>
"""


def apply():
    st.markdown(_CSS, unsafe_allow_html=True)
