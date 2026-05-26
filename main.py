"""
SimuScale - AI Decision Engine
A Streamlit app that uses NVIDIA's hosted Llama-3.3-70b to simulate three
financial scenarios for a business decision and return a concrete recommendation.

Run with:
    export NVIDIA_API_KEY="nvapi-..."
    streamlit run app.py
"""

import os
import re
import time
from datetime import datetime

import streamlit as st
from openai import OpenAI, APIError, APIConnectionError, RateLimitError


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SimuScale – AI Decision Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT  (validated up-front so failures are obvious)
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("NVIDIA_API_KEY")

if not API_KEY:
    st.error(
        "🔑 **Missing API key.** Set the `NVIDIA_API_KEY` environment variable "
        "before launching the app.\n\n"
        "```bash\nexport NVIDIA_API_KEY=\"nvapi-...\"\nstreamlit run app.py\n```"
    )
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "screen": "input",          # "input" | "results"
    "results": None,            # dict from parse_response()
    "decision_text": "",
    "business_text": "",
    "industry": "E-commerce",
    "revenue": "Under $5K/mo",
    "team_size": "Solo founder",
    "time_horizon": "6 months",
    "risk_label": "Balanced",
    "history": [],              # list of past simulations
    "last_context": None,       # the inputs used for the most recent run
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

INDUSTRIES = [
    "E-commerce", "Retail", "Food & Beverage", "SaaS / Software",
    "Consulting / Agency", "Healthcare", "Real Estate", "Manufacturing",
    "Education", "Fitness & Wellness", "Finance / Fintech",
    "Marketing / Advertising", "Other",
]

REVENUE_RANGES = [
    "Pre-revenue", "Under $5K/mo", "$5K–$20K/mo", "$20K–$50K/mo",
    "$50K–$100K/mo", "$100K–$500K/mo", "Over $500K/mo",
]

TEAM_SIZES = [
    "Solo founder", "2–5 people", "6–15 people", "16–50 people", "50+ people",
]

TIME_HORIZONS = ["3 months", "6 months", "1 year", "2–3 years"]

RISK_LABELS = [
    "Very Conservative", "Conservative", "Balanced",
    "Aggressive", "Very Aggressive",
]


# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
}

/* hide chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* canvas */
.stApp { background: #f1f5f9 !important; min-height: 100vh; }

.block-container {
    padding: 72px 20px 0 20px !important;
    max-width: 1100px !important;
    background: transparent !important;
}

/* floating card */
[data-testid="stMainBlockContainer"] {
    background: #ffffff !important;
    border-radius: 20px !important;
    box-shadow:
        0 10px 25px -5px rgba(0,0,0,0.05),
        0 8px 10px -6px rgba(0,0,0,0.05) !important;
    max-width: 1100px !important;
    margin: 0 auto 48px auto !important;
    padding: 0 36px 40px 36px !important;
    overflow: visible !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
div[data-testid="stVerticalBlock"] { background: transparent !important; }

/* fixed nav */
.simuscale-nav {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 40px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 999;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02), 0 1px 2px rgba(0,0,0,0.04);
}
.nav-left { display: flex; align-items: center; gap: 12px; }
.nav-logo { display: flex; align-items: center; gap: 8px; }
.nav-icon-wrap {
    height: 28px; width: 28px;
    border-radius: 7px;
    background: #0ea5e9;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.nav-icon-wrap svg { width: 16px; height: 16px; }
.nav-name { font-weight: 700; font-size: 15px; color: #0f172a; letter-spacing: -0.02em; }
.nav-divider { width: 1px; height: 18px; background: #e2e8f0; }
.nav-product { font-size: 13px; color: #94a3b8; font-weight: 500; }
.nav-badge {
    font-size: 11px; font-weight: 600;
    color: #0284c7; background: #e0f2fe;
    border: 1px solid #bae6fd;
    border-radius: 6px; padding: 3px 9px;
    letter-spacing: 0.02em;
}

/* hero */
.hero-wrap { padding: 32px 0 28px; max-width: 640px; }
.hero-label {
    font-size: 11px; font-weight: 700;
    color: #0ea5e9; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 10px;
}
.hero-title {
    font-size: 2.25rem; font-weight: 800;
    letter-spacing: -0.03em; line-height: 1.15;
    color: #0f172a; margin: 0 0 12px;
}
.hero-sub { font-size: 15px; color: #64748b; line-height: 1.65; margin: 0; }

/* section labels */
.section-label {
    font-size: 0.75rem; font-weight: 700;
    color: #94a3b8; letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 24px 0 12px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #f1f5f9; }

/* widget labels */
div[data-testid="stTextArea"] label p,
div[data-testid="stTextInput"] label p,
div[data-testid="stSelectbox"] label p,
div[data-testid="stSlider"] label p {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 4px !important;
}

/* inputs */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    padding: 12px 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    color: #1e293b !important;
    background: #fafbfc !important;
    line-height: 1.6 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stTextArea"] textarea { resize: none !important; }
div[data-testid="stTextArea"] textarea:hover,
div[data-testid="stTextInput"] input:hover {
    border-color: #94a3b8 !important;
    box-shadow: 0 2px 8px rgba(14,165,233,0.08) !important;
}
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #0ea5e9 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 4px rgba(14,165,233,0.12), 0 2px 8px rgba(14,165,233,0.08) !important;
    outline: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder { color: #c8d5e0 !important; }

/* selectbox */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #fafbfc !important;
    font-size: 14px !important;
    color: #1e293b !important;
    padding: 2px 4px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stSelectbox"] > div:hover { border-color: #94a3b8 !important; }
div[data-testid="stSelectbox"] > div:focus-within {
    border-color: #0ea5e9 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 4px rgba(14,165,233,0.12) !important;
}

/* slider */
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: #0ea5e9 !important; }
div[data-testid="stSlider"] [role="slider"] {
    background: #0ea5e9 !important;
    box-shadow: 0 2px 8px rgba(14,165,233,0.4) !important;
}

/* primary button */
div[data-testid="stButton"] > button {
    background: #0ea5e9 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 15px 28px !important;
    border-radius: 12px !important;
    border: none !important;
    width: 100% !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 10px rgba(14,165,233,0.25), 0 1px 3px rgba(14,165,233,0.15) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    background: #0284c7 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(14,165,233,0.35), 0 3px 8px rgba(14,165,233,0.15) !important;
}
div[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* secondary / ghost button */
div[data-testid="stButton"].secondary > button,
div[data-testid="stButton"] > button[kind="secondary"] {
    background: transparent !important;
    color: #475569 !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
    transform: none !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* result cards */
.outcome-card, .outcome-card-featured {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px;
    display: flex; flex-direction: column; gap: 12px;
    border: 1.5px solid #f1f5f9;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02), 0 1px 3px rgba(0,0,0,0.03);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
    margin-top: 12px;
}
.outcome-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.09), 0 4px 10px rgba(0,0,0,0.05);
}
.card-conservative { border-left: 4px solid #ef4444; }
.card-expected    { border-left: 4px solid #0ea5e9; position: relative; }
.card-aggressive  { border-left: 4px solid #10b981; }

.outcome-card-featured {
    border: 1.5px solid #bae6fd !important;
    box-shadow: 0 6px 24px rgba(14,165,233,0.12), 0 2px 6px rgba(14,165,233,0.06) !important;
    margin-top: 12px;
    position: relative;
}
.outcome-card-featured:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 36px rgba(14,165,233,0.18), 0 5px 12px rgba(14,165,233,0.1) !important;
}

.most-likely-pill {
    position: absolute;
    top: -11px; left: 18px;
    background: #0ea5e9;
    color: white;
    font-size: 10px; font-weight: 700;
    padding: 4px 12px;
    border-radius: 999px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    white-space: nowrap;
    box-shadow: 0 2px 6px rgba(14,165,233,0.3);
}

.card-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.card-title { font-weight: 700; font-size: 14px; color: #0f172a; }
.badge {
    font-size: 10px; font-weight: 700;
    padding: 3px 9px; border-radius: 20px;
    text-transform: uppercase; letter-spacing: 0.05em;
    white-space: nowrap; flex-shrink: 0;
}
.badge-low  { background: #fff1f2; color: #dc2626; border: 1px solid #fee2e2; }
.badge-mid  { background: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; }
.badge-high { background: #f0fdf4; color: #16a34a; border: 1px solid #dcfce7; }

.metric-stat {
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 17px;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.25;
}
.metric-low  { background: #fff1f2; color: #dc2626; }
.metric-mid  { background: #e0f2fe; color: #0284c7; }
.metric-high { background: #f0fdf4; color: #16a34a; }

.card-body { font-size: 13px; color: #475569; line-height: 1.7; flex-grow: 1; margin: 0; }

/* recommendation */
.rec-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1.5px solid #bae6fd;
    border-left: 4px solid #0ea5e9;
    border-radius: 16px;
    padding: 22px 26px;
    display: flex; gap: 16px;
    margin: 20px 0 0;
    transition: box-shadow 0.3s ease;
}
.rec-box:hover { box-shadow: 0 4px 16px rgba(14,165,233,0.12); }
.rec-icon {
    font-size: 22px; flex-shrink: 0; margin-top: 1px;
    color: #0ea5e9; font-weight: 700;
}
.rec-label {
    font-size: 0.7rem; font-weight: 700;
    color: #0284c7; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 6px;
}
.rec-body { font-size: 14.5px; color: #0f172a; line-height: 1.7; margin: 0; font-weight: 500; }

/* results header */
.results-title {
    font-size: 1.6rem; font-weight: 800;
    color: #0f172a; letter-spacing: -0.02em;
    text-align: center; margin: 8px 0 6px;
}
.results-sub { font-size: 14px; color: #64748b; text-align: center; margin-bottom: 6px; }
.results-sub strong { color: #0284c7; font-weight: 600; }

.context-chips { display: flex; flex-wrap: wrap; justify-content: center; gap: 6px; margin: 14px 0 4px; }
.chip {
    font-size: 11px; font-weight: 500;
    color: #475569; background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 20px; padding: 4px 11px;
}

/* history */
.history-header {
    font-size: 0.75rem; font-weight: 700;
    color: #94a3b8; letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 36px 0 14px;
    display: flex; align-items: center; gap: 10px;
}
.history-header::before, .history-header::after {
    content: ''; flex: 1; height: 1px; background: #f1f5f9;
}
.history-card {
    background: #fafbfc;
    border: 1px solid #f1f5f9;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.history-decision { font-weight: 600; color: #1e293b; margin-bottom: 6px; font-size: 14px; }
.history-meta { font-size: 11px; color: #94a3b8; margin-bottom: 10px; }
.history-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
.history-cell {
    background: #ffffff;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid #f1f5f9;
}
.history-cell-label {
    font-size: 9px; font-weight: 700;
    color: #94a3b8; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 3px;
}
.history-cell-stat {
    font-size: 12px; font-weight: 700;
    color: #334155; letter-spacing: -0.01em;
}

/* misc */
div[data-testid="stSpinner"] p { color: #0ea5e9 !important; font-weight: 600 !important; font-size: 14px !important; }
div[data-testid="stAlert"] { border-radius: 12px !important; }

/* tighten gaps */
div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] {
    margin-bottom: 0.75rem !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    padding: 0 6px !important;
}

/* mobile */
@media (max-width: 768px) {
    .block-container { padding: 68px 0 0 0 !important; }
    [data-testid="stMainBlockContainer"] {
        border-radius: 16px !important;
        margin: 0 0.75rem 32px !important;
        padding: 0 1.25rem 32px !important;
    }
    .simuscale-nav { padding: 0 1.25rem; }
    .nav-product, .nav-divider { display: none; }
    .hero-wrap { padding: 24px 0 20px; }
    .hero-title { font-size: 1.65rem !important; }
    .hero-sub { font-size: 14px !important; }

    div[data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
        padding: 0 !important;
    }
    .metric-stat { font-size: 15px !important; }
    .results-title { font-size: 1.3rem !important; }
    .history-grid { grid-template-columns: 1fr !important; }
}
@media (max-width: 480px) {
    .nav-badge { display: none; }
    [data-testid="stMainBlockContainer"] {
        border-radius: 12px !important;
        margin: 0 0.5rem 24px !important;
        padding: 0 1rem 24px !important;
    }
    .hero-title { font-size: 1.4rem !important; }
    .rec-box { flex-direction: column; gap: 8px; padding: 18px; }
}
</style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPT + CALL
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are SimuScale, a hard-numbers business simulation engine. Be specific – never vague.
Output EXACTLY 7 blocks separated by "|". No extra text, no markdown, no headers, no labels.
Format (6 pipes, 7 blocks): [c_stat]|[c_body]|[e_stat]|[e_body]|[a_stat]|[a_body]|[rec]

Rules for each block:
- c_stat / e_stat / a_stat : ONE metric headline, max 6 words. MUST contain a real number ($ or %).
  Examples: "+$3.2K/mo net margin", "-18% churn within 60 days", "2.8X ROI by month 6", "$22K cash burn risk"
- c_body / e_body / a_body : EXACTLY 2 SHORT sentences.
  Sentence 1: specific projected outcome with numbers tied to the stated revenue range and team size.
  Sentence 2: the single biggest risk OR accelerator for this scenario.
  No filler, no padding.
- rec : EXACTLY 2 sentences.
  Sentence 1: the single most important action to take right now with a specific number (budget, timeline, or metric).
  Sentence 2: the key metric to watch to know if it's working.

Critical: scale ALL dollar figures and percentages to fit the given revenue range. A pre-revenue solo founder and a $100K/mo team get very different numbers. Never write generic advice."""


def parse_response(raw: str) -> dict:
    """Parse the model's pipe-delimited response into a structured dict."""
    raw = (raw or "").strip()
    parts = [p.strip() for p in raw.split("|")]

    if len(parts) >= 7:
        return {
            "conservative_stat": parts[0],
            "conservative":      parts[1],
            "expected_stat":     parts[2],
            "expected":          parts[3],
            "aggressive_stat":   parts[4],
            "aggressive":        parts[5],
            "recommendation":    " ".join(parts[6:]).strip(),
        }

    # Fallback: pad missing blocks so the UI still renders.
    while len(parts) < 7:
        parts.append("—")
    return {
        "conservative_stat": parts[0] or "—",
        "conservative":      parts[1] or "Insufficient data returned by the model.",
        "expected_stat":     parts[2] or "—",
        "expected":          parts[3] or "Insufficient data returned by the model.",
        "aggressive_stat":   parts[4] or "—",
        "aggressive":        parts[5] or "Insufficient data returned by the model.",
        "recommendation":    parts[6] or "Re-run the simulation for a sharper recommendation.",
    }


def run_simulation(business, decision, industry, revenue, team_size,
                   time_horizon, risk_label) -> dict:
    """Call NVIDIA-hosted Llama and return parsed result."""
    user_prompt = (
        f"Business: {business}\n"
        f"Industry: {industry}\n"
        f"Monthly Revenue: {revenue}\n"
        f"Team Size: {team_size}\n"
        f"Time Horizon: {time_horizon}\n"
        f"Risk Tolerance: {risk_label}\n"
        f"Decision: {decision}\n\n"
        f"Output the 7 blocks now:"
    )

    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=520,
    )
    return parse_response(response.choices[0].message.content)


def risk_to_label(value: int) -> str:
    """Map slider 1–5 to risk label."""
    return RISK_LABELS[max(0, min(4, value - 1))]


# ─────────────────────────────────────────────────────────────────────────────
# NAV
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="simuscale-nav">
    <div class="nav-left">
        <div class="nav-logo">
            <div class="nav-icon-wrap">
                <svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M2 12L6 7L9 10L14 4" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="14" cy="4" r="1.2" fill="white"/>
                </svg>
            </div>
            <span class="nav-name">SimuScale</span>
            <span class="nav-divider"></span>
            <span class="nav-product">Decision Analytics</span>
        </div>
    </div>
    <span class="nav-badge">Powered by NVIDIA</span>
</div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def render_input_screen():
    st.markdown(
        """
<div class="hero-wrap">
    <div class="hero-label">Decision Intelligence Platform</div>
    <h1 class="hero-title">Model your next business decision.</h1>
    <p class="hero-sub">Enter your context and the decision you're weighing.
    SimuScale projects three financial scenarios — conservative, expected, and aggressive —
    and gives you one concrete action to take next.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Decision block ───
    st.markdown('<div class="section-label">The decision</div>', unsafe_allow_html=True)

    st.session_state.business_text = st.text_area(
        "Your business",
        value=st.session_state.business_text,
        placeholder="e.g. DTC skincare brand selling on Shopify. Mostly Instagram traffic.",
        height=80,
        key="business_input",
    )

    st.session_state.decision_text = st.text_area(
        "The decision you're considering",
        value=st.session_state.decision_text,
        placeholder="e.g. Spend $5K/month on Meta ads for the next 3 months to test scaling paid acquisition.",
        height=110,
        key="decision_input",
    )

    # ─── Context block ───
    st.markdown('<div class="section-label">Context</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.industry = st.selectbox(
            "Industry", INDUSTRIES,
            index=INDUSTRIES.index(st.session_state.industry),
        )
        st.session_state.team_size = st.selectbox(
            "Team size", TEAM_SIZES,
            index=TEAM_SIZES.index(st.session_state.team_size),
        )
    with c2:
        st.session_state.revenue = st.selectbox(
            "Monthly revenue", REVENUE_RANGES,
            index=REVENUE_RANGES.index(st.session_state.revenue),
        )
        st.session_state.time_horizon = st.selectbox(
            "Time horizon", TIME_HORIZONS,
            index=TIME_HORIZONS.index(st.session_state.time_horizon),
        )

    # risk slider
    risk_idx = st.slider(
        f"Risk tolerance — **{st.session_state.risk_label}**",
        min_value=1, max_value=5,
        value=RISK_LABELS.index(st.session_state.risk_label) + 1,
        step=1,
    )
    st.session_state.risk_label = risk_to_label(risk_idx)

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    # ─── Run button ───
    run = st.button("⚡  Run Simulation", key="run_sim", type="primary")

    if run:
        business = st.session_state.business_text.strip()
        decision = st.session_state.decision_text.strip()

        if not decision:
            st.warning("Please describe the decision you're considering.")
            return
        if len(decision) < 12:
            st.warning("Add a little more detail about the decision (at least a sentence).")
            return
        if not business:
            business = "(not specified)"

        try:
            with st.spinner("Simulating outcomes across three scenarios…"):
                result = run_simulation(
                    business=business,
                    decision=decision,
                    industry=st.session_state.industry,
                    revenue=st.session_state.revenue,
                    team_size=st.session_state.team_size,
                    time_horizon=st.session_state.time_horizon,
                    risk_label=st.session_state.risk_label,
                )

            st.session_state.results = result
            st.session_state.last_context = {
                "business":     business,
                "decision":     decision,
                "industry":     st.session_state.industry,
                "revenue":      st.session_state.revenue,
                "team_size":    st.session_state.team_size,
                "time_horizon": st.session_state.time_horizon,
                "risk_label":   st.session_state.risk_label,
                "timestamp":    datetime.now().strftime("%b %d, %Y · %I:%M %p"),
            }
            # add to history (cap at 10)
            st.session_state.history.insert(
                0,
                {**st.session_state.last_context, "result": result},
            )
            st.session_state.history = st.session_state.history[:10]
            st.session_state.screen = "results"
            st.rerun()

        except RateLimitError:
            st.error("⏱️ Rate limit hit. Wait a few seconds and try again.")
        except APIConnectionError:
            st.error("🌐 Couldn't reach the NVIDIA API. Check your connection and try again.")
        except APIError as e:
            st.error(f"⚠️ API error: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    # ─── History preview on input screen ───
    if st.session_state.history:
        st.markdown(
            '<div class="history-header">Recent simulations</div>',
            unsafe_allow_html=True,
        )
        for h in st.session_state.history[:3]:
            r = h["result"]
            st.markdown(
                f"""
<div class="history-card">
    <div class="history-decision">{_escape(h["decision"][:140])}{"…" if len(h["decision"]) > 140 else ""}</div>
    <div class="history-meta">{_escape(h["industry"])} · {_escape(h["revenue"])} · {_escape(h["timestamp"])}</div>
    <div class="history-grid">
        <div class="history-cell">
            <div class="history-cell-label">Conservative</div>
            <div class="history-cell-stat">{_escape(r["conservative_stat"])}</div>
        </div>
        <div class="history-cell">
            <div class="history-cell-label">Expected</div>
            <div class="history-cell-stat">{_escape(r["expected_stat"])}</div>
        </div>
        <div class="history-cell">
            <div class="history-cell-label">Aggressive</div>
            <div class="history-cell-stat">{_escape(r["aggressive_stat"])}</div>
        </div>
    </div>
</div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def render_results_screen():
    result = st.session_state.results
    ctx = st.session_state.last_context or {}

    st.markdown(
        f"""
<div style="padding-top: 24px;">
    <h2 class="results-title">Three scenarios for your decision</h2>
    <p class="results-sub">Based on: <strong>{_escape((ctx.get("decision") or "")[:120])}{"…" if len(ctx.get("decision") or "") > 120 else ""}</strong></p>
    <div class="context-chips">
        <span class="chip">🏢 {_escape(ctx.get("industry", ""))}</span>
        <span class="chip">💰 {_escape(ctx.get("revenue", ""))}</span>
        <span class="chip">👥 {_escape(ctx.get("team_size", ""))}</span>
        <span class="chip">📅 {_escape(ctx.get("time_horizon", ""))}</span>
        <span class="chip">⚖️ {_escape(ctx.get("risk_label", ""))}</span>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # three scenario cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
<div class="outcome-card card-conservative">
    <div class="card-top">
        <span class="card-title">Conservative</span>
        <span class="badge badge-low">Downside</span>
    </div>
    <div class="metric-stat metric-low">{_escape(result["conservative_stat"])}</div>
    <p class="card-body">{_escape(result["conservative"])}</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
<div class="outcome-card-featured">
    <span class="most-likely-pill">Most likely</span>
    <div class="card-top">
        <span class="card-title">Expected</span>
        <span class="badge badge-mid">Base case</span>
    </div>
    <div class="metric-stat metric-mid">{_escape(result["expected_stat"])}</div>
    <p class="card-body">{_escape(result["expected"])}</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
<div class="outcome-card card-aggressive">
    <div class="card-top">
        <span class="card-title">Aggressive</span>
        <span class="badge badge-high">Upside</span>
    </div>
    <div class="metric-stat metric-high">{_escape(result["aggressive_stat"])}</div>
    <p class="card-body">{_escape(result["aggressive"])}</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    # recommendation
    st.markdown(
        f"""
<div class="rec-box">
    <div class="rec-icon">→</div>
    <div>
        <div class="rec-label">Recommended next move</div>
        <p class="rec-body">{_escape(result["recommendation"])}</p>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # actions
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    a1, a2 = st.columns([1, 1])
    with a1:
        if st.button("← New simulation", key="back_btn", type="secondary"):
            st.session_state.screen = "input"
            st.session_state.results = None
            st.rerun()
    with a2:
        if st.button("🔄  Re-run with same inputs", key="rerun_btn"):
            try:
                with st.spinner("Re-running…"):
                    result = run_simulation(
                        business=ctx.get("business", "(not specified)"),
                        decision=ctx.get("decision", ""),
                        industry=ctx.get("industry", "Other"),
                        revenue=ctx.get("revenue", "Pre-revenue"),
                        team_size=ctx.get("team_size", "Solo founder"),
                        time_horizon=ctx.get("time_horizon", "6 months"),
                        risk_label=ctx.get("risk_label", "Balanced"),
                    )
                st.session_state.results = result
                st.session_state.history.insert(
                    0,
                    {**ctx,
                     "timestamp": datetime.now().strftime("%b %d, %Y · %I:%M %p"),
                     "result": result},
                )
                st.session_state.history = st.session_state.history[:10]
                st.rerun()
            except Exception as e:
                st.error(f"Re-run failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _escape(s: str) -> str:
    """Minimal HTML escape so model output can't break the layout."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.screen == "results" and st.session_state.results:
    render_results_screen()
else:
    render_input_screen()
