import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CrisisVision AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# PREMIUM DARK THEME CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* ── Tabs ─────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.8) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.25) !important;
    color: #a5b4fc !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── Hero Header ─────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #1a0a2e 0%, #16213e 40%, #0f3460 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
    background: radial-gradient(ellipse at center,rgba(99,102,241,0.08) 0%,transparent 60%);
    pointer-events:none;
}
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg,#ff4757,#ff6b81,#ffa502);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1.1;
}
.hero-subtitle { font-size: 0.98rem; color: #94a3b8; margin-top: 0.5rem; }
.hero-badge {
    display:inline-block; background:rgba(99,102,241,0.15);
    border:1px solid rgba(99,102,241,0.4); color:#818cf8;
    padding:3px 12px; border-radius:20px; font-size:0.75rem;
    font-weight:600; letter-spacing:0.05em; margin-bottom:0.8rem;
}

/* ── Cards ───────────────────────────── */
.card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.4rem;
    backdrop-filter: blur(12px);
    animation: fadeInUp 0.4s ease forwards;
}
.card-danger  { border-color:rgba(239,68,68,0.4);  background:rgba(239,68,68,0.05);  }
.card-warning { border-color:rgba(245,158,11,0.4); background:rgba(245,158,11,0.05); }

/* ── Alert Banners ───────────────────── */
.alert-critical { background:linear-gradient(135deg,#7f1d1d,#991b1b); border:1px solid #ef4444; color:#fca5a5; border-radius:12px; padding:1.1rem 1.6rem; font-size:1.1rem; font-weight:700; animation:pulseGlow 2s infinite; display:flex; align-items:center; gap:12px; }
.alert-high     { background:linear-gradient(135deg,#78350f,#92400e); border:1px solid #f59e0b; color:#fcd34d; border-radius:12px; padding:1.1rem 1.6rem; font-size:1.1rem; font-weight:700; display:flex; align-items:center; gap:12px; }
.alert-medium   { background:linear-gradient(135deg,#1a2e1a,#14532d); border:1px solid #10b981; color:#6ee7b7; border-radius:12px; padding:1.1rem 1.6rem; font-size:1.1rem; font-weight:700; display:flex; align-items:center; gap:12px; }
.alert-low      { background:rgba(99,102,241,0.1); border:1px solid #6366f1; color:#a5b4fc; border-radius:12px; padding:1.1rem 1.6rem; font-size:1.1rem; font-weight:700; display:flex; align-items:center; gap:12px; }

/* ── Confidence Bar ──────────────────── */
.conf-label { font-size:0.78rem; color:#94a3b8; font-weight:500; margin-bottom:6px; display:flex; justify-content:space-between; }
.conf-track { background:rgba(30,41,59,0.8); border-radius:100px; height:10px; width:100%; overflow:hidden; border:1px solid rgba(99,102,241,0.2); }
.conf-fill-critical { background:linear-gradient(90deg,#ef4444,#dc2626); height:100%; border-radius:100px; }
.conf-fill-high     { background:linear-gradient(90deg,#f59e0b,#d97706); height:100%; border-radius:100px; }
.conf-fill-medium   { background:linear-gradient(90deg,#10b981,#059669); height:100%; border-radius:100px; }
.conf-fill-low      { background:linear-gradient(90deg,#6366f1,#4f46e5); height:100%; border-radius:100px; }

/* ── Location Tags ───────────────────── */
.loc-tag { display:inline-block; background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.4); color:#c7d2fe; padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:500; margin:3px; }

/* ── AI Summary ──────────────────────── */
.ai-summary { background:rgba(30,41,59,0.6); border-left:4px solid #6366f1; border-radius:0 12px 12px 0; padding:1.1rem 1.4rem; color:#cbd5e1; font-size:0.9rem; line-height:1.7; white-space:pre-wrap; }

/* ── Section Labels ──────────────────── */
.section-label { font-size:0.72rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:#6366f1; margin-bottom:0.7rem; display:flex; align-items:center; gap:8px; }
.section-label::after { content:''; flex:1; height:1px; background:rgba(99,102,241,0.2); }

/* ── News Feed Card ──────────────────── */
.news-card {
    background:rgba(15,23,42,0.8); border:1px solid rgba(99,102,241,0.18);
    border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:0.75rem;
    display:grid; grid-template-columns:1fr auto; gap:12px; align-items:start;
    transition:border-color 0.2s;
}
.news-card:hover { border-color:rgba(99,102,241,0.4); }
.news-title { color:#e2e8f0; font-size:0.9rem; font-weight:600; line-height:1.45; margin-bottom:6px; }
.news-meta  { color:#475569; font-size:0.76rem; display:flex; gap:10px; align-items:center; }
.news-domain { background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25); color:#818cf8; padding:2px 9px; border-radius:10px; font-size:0.72rem; }

/* ── Batch result row ────────────────── */
.batch-row {
    display:grid; grid-template-columns:1fr 130px 80px 110px;
    gap:10px; padding:9px 12px; border-radius:8px; font-size:0.82rem;
    color:#94a3b8; border-bottom:1px solid rgba(99,102,241,0.08);
}
.batch-row:hover { background:rgba(99,102,241,0.05); }
.batch-header { font-weight:700; color:#64748b; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.08em; }

/* ── History ─────────────────────────── */
.hist-row { display:grid; grid-template-columns:130px 1fr 100px 1fr; gap:12px; padding:9px 14px; border-radius:8px; font-size:0.81rem; color:#94a3b8; border-bottom:1px solid rgba(99,102,241,0.08); }
.hist-row:hover { background:rgba(99,102,241,0.05); }
.hist-header { font-weight:700; color:#64748b; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.08em; }

/* ── Sidebar ─────────────────────────── */
[data-testid="stSidebar"] { background:rgba(10,14,26,0.95) !important; border-right:1px solid rgba(99,102,241,0.2) !important; }
.sidebar-logo { font-size:1.5rem; font-weight:800; background:linear-gradient(135deg,#ff4757,#ffa502); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.sidebar-section { background:rgba(15,23,42,0.7); border:1px solid rgba(99,102,241,0.15); border-radius:12px; padding:0.9rem 1rem; margin-bottom:0.9rem; }
.sidebar-section-title { font-size:0.7rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#6366f1; margin-bottom:0.6rem; }
.class-chip { display:inline-block; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25); color:#a5b4fc; padding:3px 9px; border-radius:14px; font-size:0.72rem; margin:2px; }

/* ── Inputs ──────────────────────────── */
textarea, .stTextArea textarea { background:rgba(15,23,42,0.9) !important; border:1px solid rgba(99,102,241,0.3) !important; border-radius:12px !important; color:#e2e8f0 !important; font-family:'Inter',sans-serif !important; }

/* ── Buttons ─────────────────────────── */
.stButton > button { background:linear-gradient(135deg,#ef4444,#dc2626) !important; color:white !important; border:none !important; border-radius:12px !important; padding:0.65rem 2rem !important; font-size:0.95rem !important; font-weight:700 !important; font-family:'Inter',sans-serif !important; width:100% !important; box-shadow:0 4px 20px rgba(239,68,68,0.3) !important; transition:all 0.2s ease !important; }
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 30px rgba(239,68,68,0.45) !important; }

/* ── Animations ──────────────────────── */
@keyframes fadeInUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
@keyframes pulseGlow { 0%,100%{box-shadow:0 0 12px rgba(239,68,68,0.3)} 50%{box-shadow:0 0 28px rgba(239,68,68,0.6)} }

hr { border-color:rgba(99,102,241,0.15) !important; }
.streamlit-expanderHeader { background:rgba(15,23,42,0.7) !important; border-radius:10px !important; color:#94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
for key, default in [
    ("history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_alert_html(severity, disaster):
    icons = {"Critical":"🔴","High":"🟠","Medium":"🟢","Low":"🔵"}
    css   = {"Critical":"alert-critical","High":"alert-high","Medium":"alert-medium","Low":"alert-low"}
    return f"""
    <div class="{css.get(severity,'alert-low')}">
      <span style="font-size:1.6rem">{icons.get(severity,'⚪')}</span>
      <div>
        <div style="font-size:0.72rem;opacity:0.7;letter-spacing:0.1em;text-transform:uppercase">Alert Level</div>
        <div style="font-size:1.3rem">{severity} — {disaster.replace('_',' ').title()}</div>
      </div>
    </div>"""

def conf_bar(confidence, severity):
    pct = int(confidence * 100)
    return f"""
    <div class="conf-label"><span>Confidence</span><span style="color:#e2e8f0;font-weight:700">{pct}%</span></div>
    <div class="conf-track"><div class="conf-fill-{severity.lower()}" style="width:{pct}%"></div></div>"""

def loc_chips(locations):
    if isinstance(locations, list):
        return "".join([f'<span class="loc-tag">📍 {l}</span>' for l in locations])
    elif locations == "Unknown":
        return '<span class="loc-tag" style="color:#475569">📍 No location detected</span>'
    return f'<span class="loc-tag">📍 {locations}</span>'

def fetch_classes():
    try:
        r = requests.get(f"{BACKEND_URL}/classes", timeout=3)
        return r.json().get("disaster_types", []) if r.status_code == 200 else []
    except: return []

def fetch_health():
    try: return requests.get(f"{BACKEND_URL}/health", timeout=3).status_code == 200
    except: return False

SEV_COLOR = {"Critical":"#ef4444","High":"#f59e0b","Medium":"#10b981","Low":"#6366f1"}


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🚨 CrisisVision AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:0.8rem;margin-bottom:1.2rem">Multimodal Crisis Detection v2.0</p>', unsafe_allow_html=True)

    is_alive = fetch_health()
    sc, st_txt = ("#10b981","Online") if is_alive else ("#ef4444","Offline")
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-section-title">🔌 Backend Status</div>
        <div style="display:flex;align-items:center;gap:8px">
            <div style="width:9px;height:9px;border-radius:50%;background:{sc};box-shadow:0 0 8px {sc}"></div>
            <span style="color:{sc};font-weight:600;font-size:0.86rem">{st_txt}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">🧠 Model Stack</div>
        <div style="color:#94a3b8;font-size:0.81rem;line-height:1.9">
            <b style="color:#e2e8f0">Text:</b> BERT-base-uncased<br>
            <b style="color:#e2e8f0">Vision:</b> ResNet-50<br>
            <b style="color:#e2e8f0">NER:</b> XLM-RoBERTa<br>
            <b style="color:#e2e8f0">GenAI:</b> Llama 3.3 70B (Groq)
        </div>
    </div>""", unsafe_allow_html=True)

    classes = fetch_classes()
    if classes:
        chips = "".join([f'<span class="class-chip">{c.replace("_"," ").title()}</span>' for c in classes])
        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">📂 Crisis Types</div>
            {chips}
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">📖 How to Use</div>
        <div style="color:#94a3b8;font-size:0.8rem;line-height:1.8">
            🔍 <b style="color:#e2e8f0">Single</b> — Paste a tweet + upload image<br>
            📊 <b style="color:#e2e8f0">Batch</b> — Upload a CSV of tweets
        </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()


# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🤖 Multimodal AI · BERT + ResNet-50 · Llama 3.3 70B</div>
    <div class="hero-title">AI-Powered Crisis Detection</div>
    <div class="hero-subtitle">
        Analyze individual tweets or run bulk batch analysis on CSV files.
        Powered by BERT + ResNet-50 multimodal fusion with Groq AI emergency reports.
    </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab_single, tab_batch = st.tabs([
    "🔍  Single Analysis",
    "📊  Batch Analysis",
])


# ═══════════════════════════════════════════
# TAB 1 — SINGLE ANALYSIS
# ═══════════════════════════════════════════
with tab_single:
    col_in, col_prev = st.columns([3, 2], gap="large")

    with col_in:
        st.markdown('<div class="section-label">📝 Tweet Input</div>', unsafe_allow_html=True)
        tweet_text = st.text_area("tweet_input",
            placeholder="e.g. Massive flooding in Kerala. Roads submerged, people stranded. Need urgent evacuation! #KeralaFloods",
            height=150, label_visibility="collapsed")

        st.markdown('<div class="section-label" style="margin-top:1rem">📷 Disaster Image</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("image_upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze Crisis", use_container_width=True, key="single_btn")

    with col_prev:
        st.markdown('<div class="section-label">🖼️ Preview</div>', unsafe_allow_html=True)
        if uploaded_image:
            st.image(uploaded_image, use_column_width=True, caption=f"📎 {uploaded_image.name}")
        else:
            st.markdown("""
            <div style="height:240px;border:2px dashed rgba(99,102,241,0.25);border-radius:14px;
                        display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px">
                <div style="font-size:2rem">🖼️</div>
                <div style="color:#475569;font-size:0.82rem">Upload an image to preview</div>
            </div>""", unsafe_allow_html=True)

    if analyze_btn:
        if not tweet_text or not uploaded_image:
            st.markdown('<div class="card card-warning" style="margin-top:1rem"><b style="color:#fcd34d">⚠️ Missing Input</b><p style="color:#94a3b8;margin:0.4rem 0 0">Please provide both tweet text and an image.</p></div>', unsafe_allow_html=True)
        else:
            with st.spinner("🔄 Running multimodal analysis..."):
                try:
                    uploaded_image.seek(0)
                    resp = requests.post(f"{BACKEND_URL}/predict/",
                        files={"image": (uploaded_image.name, uploaded_image, uploaded_image.type)},
                        data={"text": tweet_text}, timeout=60)

                    if resp.status_code == 200:
                        r = resp.json()
                        disaster, confidence = r["disaster_type"], r["confidence"]
                        locations, summary = r["locations_detected"], r["ai_summary"]
                        severity = r.get("severity_level", "Low")

                        st.session_state.history.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "disaster": disaster, "confidence": f"{confidence*100:.1f}%",
                            "severity": severity,
                            "locations": locations if isinstance(locations, str) else ", ".join(locations)
                        })

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-label">🚨 Results</div>', unsafe_allow_html=True)
                        st.markdown(get_alert_html(severity, disaster), unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

                        ca, cb, cc = st.columns(3, gap="medium")
                        with ca:
                            st.markdown(f"""
                            <div class="card" style="text-align:center">
                                <div style="font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:#64748b">Disaster Type</div>
                                <div style="font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-top:8px">{disaster.replace('_',' ').title()}</div>
                            </div>""", unsafe_allow_html=True)
                        with cb:
                            st.markdown(f'<div class="card">{conf_bar(confidence,severity)}<div style="color:#64748b;font-size:0.76rem;margin-top:8px">Raw: <b style="color:#e2e8f0">{confidence:.6f}</b></div></div>', unsafe_allow_html=True)
                        with cc:
                            st.markdown(f'<div class="card"><div style="font-size:0.72rem;color:#64748b;font-weight:600;margin-bottom:8px">LOCATIONS</div>{loc_chips(locations)}</div>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-label">🤖 AI Emergency Report</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="card">
                            <div style="margin-bottom:0.9rem">
                                <span style="background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);border-radius:8px;padding:4px 12px;font-size:0.74rem;color:#818cf8;font-weight:600">🦙 Llama 3.3 70B · Groq</span>
                            </div>
                            <div class="ai-summary">{summary}</div>
                        </div>""", unsafe_allow_html=True)

                        with st.expander("🔎 Raw JSON Output"):
                            st.json(r)
                    else:
                        st.markdown(f'<div class="card card-danger"><b style="color:#fca5a5">❌ Error {resp.status_code}</b><p style="color:#94a3b8;margin:0.3rem 0 0">{resp.text[:300]}</p></div>', unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.markdown('<div class="card card-danger"><b style="color:#fca5a5">❌ Cannot reach backend</b><p style="color:#94a3b8;margin:0.3rem 0 0">Run: <code style="color:#818cf8">uvicorn main:app --reload</code></p></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="card card-danger"><b style="color:#fca5a5">❌ Error</b><p style="color:#94a3b8;margin:0.3rem 0 0">{e}</p></div>', unsafe_allow_html=True)

    # History
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">📋 Session History</div>', unsafe_allow_html=True)
        rows = '<div class="card"><div class="hist-row hist-header"><div>Time</div><div>Disaster</div><div>Confidence</div><div>Locations</div></div>'
        for e in reversed(st.session_state.history[-10:]):
            c = SEV_COLOR.get(e.get("severity","Low"), "#6366f1")
            rows += f'<div class="hist-row"><div style="color:#64748b">{e["timestamp"]}</div><div style="color:#e2e8f0;font-weight:600">{e["disaster"].replace("_"," ").title()}</div><div style="color:{c};font-weight:700">{e["confidence"]} <span style="color:#475569;font-size:0.7rem">{e.get("severity","")}</span></div><div style="color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{e["locations"]}</div></div>'
        rows += "</div>"
        st.markdown(rows, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ═══════════════════════════════════════════
with tab_batch:
    st.markdown("""
    <div class="card" style="margin-bottom:1.2rem">
        <div style="color:#94a3b8;font-size:0.87rem;line-height:1.7">
            📂 Upload a <b style="color:#e2e8f0">CSV file</b> with a column named <code style="color:#818cf8">text</code> containing tweet texts.<br>
            Optionally upload one <b style="color:#e2e8f0">representative image</b> (used for all tweets). Without an image a neutral placeholder is used.
        </div>
    </div>""", unsafe_allow_html=True)

    bc1, bc2 = st.columns([2, 1], gap="large")

    with bc1:
        st.markdown('<div class="section-label">📂 Upload CSV</div>', unsafe_allow_html=True)
        csv_file = st.file_uploader("csv_upload", type=["csv"], label_visibility="collapsed")
        if csv_file:
            df_preview = pd.read_csv(csv_file, on_bad_lines='skip', quotechar='"', engine='python')
            csv_file.seek(0)
            if "text" not in df_preview.columns:
                st.markdown('<div class="card card-warning"><b style="color:#fcd34d">⚠️ CSV must have a column named "text"</b></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:#64748b;font-size:0.8rem;margin-bottom:6px">{len(df_preview)} tweets found · Preview:</div>', unsafe_allow_html=True)
                st.dataframe(df_preview[["text"]].head(5), use_container_width=True, hide_index=True)

    with bc2:
        st.markdown('<div class="section-label">📷 Representative Image (optional)</div>', unsafe_allow_html=True)
        batch_image = st.file_uploader("batch_img_upload", type=["jpg","jpeg","png"], label_visibility="collapsed", key="batch_img")
        st.markdown("<br>", unsafe_allow_html=True)
        batch_btn = st.button("📊 Run Batch Analysis", use_container_width=True, key="batch_btn")

    if batch_btn:
        if not csv_file:
            st.markdown('<div class="card card-warning"><b style="color:#fcd34d">⚠️ Please upload a CSV file first.</b></div>', unsafe_allow_html=True)
        else:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, on_bad_lines='skip', quotechar='"', engine='python')
            if "text" not in df.columns:
                st.markdown('<div class="card card-warning"><b style="color:#fcd34d">⚠️ CSV must have a column named "text"</b></div>', unsafe_allow_html=True)
            else:
                tweets = df["text"].dropna().tolist()
                if len(tweets) > 50:
                    tweets = tweets[:50]
                    st.info("ℹ️ Capped at 50 tweets for performance.")

                with st.spinner(f"🔄 Analysing {len(tweets)} tweets..."):
                    try:
                        files = {}
                        if batch_image:
                            batch_image.seek(0)
                            files["image"] = (batch_image.name, batch_image, batch_image.type)

                        resp = requests.post(
                            f"{BACKEND_URL}/predict-batch/",
                            data={"texts": json.dumps(tweets)},
                            files=files if files else None,
                            timeout=300
                        )

                        if resp.status_code == 200:
                            br = resp.json()
                            agg = br["aggregate"]
                            results_list = br["results"]
                            ai_batch_sum = br.get("ai_batch_summary","")

                            # ── KPI Row ──
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown('<div class="section-label">📊 Batch Results</div>', unsafe_allow_html=True)
                            k1, k2, k3, k4 = st.columns(4, gap="small")
                            def kpi(col, label, value, color="#6366f1"):
                                col.markdown(f"""
                                <div class="card" style="text-align:center">
                                    <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em">{label}</div>
                                    <div style="font-size:1.9rem;font-weight:800;color:{color};margin-top:4px">{value}</div>
                                </div>""", unsafe_allow_html=True)
                            kpi(k1, "Tweets Analyzed", br["analysed"], "#6366f1")
                            kpi(k2, "Dominant Crisis", agg["dominant_type"].replace("_"," ").title(), "#ef4444")
                            kpi(k3, "Avg Confidence", f"{agg['avg_confidence']*100:.1f}%", "#f59e0b")
                            critical_n = agg["severity_counts"].get("Critical", 0)
                            kpi(k4, "Critical Alerts", critical_n, "#ef4444" if critical_n > 0 else "#10b981")

                            st.markdown("<br>", unsafe_allow_html=True)
                            ch1, ch2 = st.columns(2, gap="medium")

                            with ch1:
                                type_data = agg["disaster_type_counts"]
                                fig_pie = px.pie(
                                    values=list(type_data.values()),
                                    names=[k.replace("_"," ").title() for k in type_data.keys()],
                                    title="Crisis Type Distribution",
                                    color_discrete_sequence=["#ef4444","#f59e0b","#10b981","#6366f1","#ec4899"]
                                )
                                fig_pie.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(family="Inter", color="#94a3b8"),
                                    title_font=dict(color="#e2e8f0", size=14),
                                    legend=dict(bgcolor="rgba(0,0,0,0)")
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)

                            with ch2:
                                sev_data = agg["severity_counts"]
                                sev_order = ["Critical","High","Medium","Low"]
                                sev_vals  = [sev_data.get(s,0) for s in sev_order]
                                sev_colors = ["#ef4444","#f59e0b","#10b981","#6366f1"]
                                fig_bar = go.Figure(go.Bar(
                                    x=sev_order, y=sev_vals,
                                    marker_color=sev_colors,
                                    text=sev_vals, textposition="outside",
                                    textfont=dict(color="#e2e8f0")
                                ))
                                fig_bar.update_layout(
                                    title="Severity Distribution",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(family="Inter", color="#94a3b8"),
                                    title_font=dict(color="#e2e8f0", size=14),
                                    yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                                    xaxis=dict(gridcolor="rgba(0,0,0,0)")
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)

                            # ── AI Batch Summary ──
                            if ai_batch_sum:
                                st.markdown('<div class="section-label">🤖 AI Batch Intelligence Report</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class="card">
                                    <div style="margin-bottom:0.8rem"><span style="background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);border-radius:8px;padding:4px 12px;font-size:0.74rem;color:#818cf8;font-weight:600">🦙 Llama 3.3 70B · Groq · Batch Mode</span></div>
                                    <div class="ai-summary">{ai_batch_sum}</div>
                                </div>""", unsafe_allow_html=True)

                            # ── Per-tweet table ──
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown('<div class="section-label">📋 Per-Tweet Results</div>', unsafe_allow_html=True)
                            valid_rows = [r for r in results_list if "disaster_type" in r]
                            if valid_rows:
                                table_rows = '<div class="card"><div class="batch-row batch-header"><div>Tweet</div><div>Disaster Type</div><div>Confidence</div><div>Severity</div></div>'
                                for row in valid_rows:
                                    sc = SEV_COLOR.get(row["severity_level"],"#6366f1")
                                    tweet_short = row["tweet"][:80] + "..." if len(row["tweet"]) > 80 else row["tweet"]
                                    table_rows += f'<div class="batch-row"><div style="color:#94a3b8;overflow:hidden">{tweet_short}</div><div style="color:#e2e8f0;font-weight:600">{row["disaster_type"].replace("_"," ").title()}</div><div style="color:#e2e8f0">{row["confidence"]*100:.1f}%</div><div style="color:{sc};font-weight:700">{row["severity_level"]}</div></div>'
                                table_rows += "</div>"
                                st.markdown(table_rows, unsafe_allow_html=True)

                                # Download CSV
                                out_df = pd.DataFrame([{
                                    "tweet": r["tweet"],
                                    "disaster_type": r["disaster_type"],
                                    "confidence": round(r["confidence"], 4),
                                    "severity": r["severity_level"],
                                    "locations": r["locations_detected"] if isinstance(r["locations_detected"], str) else ", ".join(r["locations_detected"])
                                } for r in valid_rows])
                                st.download_button(
                                    "⬇️ Download Results CSV",
                                    data=out_df.to_csv(index=False),
                                    file_name="crisis_batch_results.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.markdown(f'<div class="card card-danger"><b style="color:#fca5a5">❌ Backend Error {resp.status_code}</b><p style="color:#94a3b8;margin:0.3rem 0 0">{resp.text[:400]}</p></div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="card card-danger"><b style="color:#fca5a5">❌ Error</b><p style="color:#94a3b8;margin:0.3rem 0 0">{e}</p></div>', unsafe_allow_html=True)

    # ── Sample CSV Download ──
    st.markdown("<br>", unsafe_allow_html=True)
    import io, csv as _csv
    _buf = io.StringIO()
    _writer = _csv.writer(_buf, quoting=_csv.QUOTE_ALL)
    _writer.writerow(["text"])
    for _t in [
        "Massive flooding in Mumbai streets. People trapped on rooftops. #MumbaiFloods",
        "Huge wildfire spreading near Los Angeles. Thousands evacuated. #CAFire",
        "Strong earthquake felt across Tokyo. Buildings damaged. #Earthquake",
        "Hurricane approaching Florida coast. Evacuation orders issued. #Hurricane",
        "Severe flood warnings in Kerala, India. Rivers overflowing banks. #Flood",
        "Forest fire raging in California hills. Air quality critical. #Wildfire",
        "Earthquake shakes buildings in Istanbul. People fleeing into streets. #Turkey",
        "Flash floods sweep cars in Phoenix. Emergency crews deployed. #Phoenix",
    ]:
        _writer.writerow([_t])
    sample_csv = _buf.getvalue()
    st.download_button("📥 Download Sample CSV", data=sample_csv, file_name="sample_crisis_tweets.csv", mime="text/csv")

