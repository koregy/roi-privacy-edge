"""Streamlit dashboard for ROI Privacy Edge."""

import json
from collections import Counter
from pathlib import Path

import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

STATS_PATH = "/tmp/roi_stats.json"
CONTROL_PATH = "/tmp/roi_control.json"
REFRESH_MS = 1000

STATE_COLORS = {
    "NORMAL": ("🟢", "#28a745", "Normal"),
    "WARNING": ("🟡", "#ffc107", "Warning"),
    "EMERGENCY": ("🔴", "#dc3545", "Emergency"),
    "UNKNOWN": ("⚪", "#6c757d", "Unknown"),
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = PROJECT_ROOT / "results"


def read_stats():
    try:
        with open(STATS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_control(updates: dict):
    try:
        with open(CONTROL_PATH) as f:
            current = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        current = {}
    current.update(updates)
    try:
        with open(CONTROL_PATH, "w") as f:
            json.dump(current, f)
    except OSError:
        pass


def detect_mode(stats):
    r = stats.get('recovery_enabled', False)
    k = stats.get('kalman_enabled', False)
    p = stats.get('predict_enabled', False)
    if not r:
        return ("🔴", "Raw")
    if not k and not p:
        return ("🟡", "Basic")
    if r and k and p:
        return ("🟢", "Full")
    return ("⚪", "Custom")


st.set_page_config(
    page_title="ROI Privacy Edge",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Custom CSS ===
st.markdown("""
<style>
    .main > div { padding-top: 0.5rem; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 0.75rem;
                border-radius: 0.5rem; border: 1px solid rgba(255,255,255,0.1); }
    .stMetric label { font-size: 0.85rem; opacity: 0.7; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; }
    section[data-testid="stSidebar"] {
        background-color: rgba(0,0,0,0.2);
        width: 220px !important;
    }
    section[data-testid="stSidebar"] > div { width: 220px !important; }
    .state-banner {
        padding: 0.75rem; border-radius: 0.5rem; text-align: center;
        color: white; font-size: 1.1rem; font-weight: 600;
        margin-bottom: 1rem;
    }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
    
    /* Tight chart spacing */
    div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] {
        margin-bottom: 0.25rem !important;
    }
    .stMarkdown { margin-bottom: 0.25rem !important; }
</style>
""", unsafe_allow_html=True)

# === Tabs ===
tab1, tab2 = st.tabs(["🔴 Live Demo", "📊 Comparison"])

# ===== Tab 1: Live Demo =====
with tab1:
    # ----- Sidebar Controls -----
    with st.sidebar:
        st.title("⚙️ Controls")
        st.caption("Live system parameters")
        
        st.divider()
        st.subheader("Network Drop")
        if st.button("Clean (0%)", use_container_width=True, key="d_clean"):
            write_control({"drop_prob": 0.0})
        if st.button("Light (15%)", use_container_width=True, key="d_light"):
            write_control({"drop_prob": 0.15})
        if st.button("Heavy (30%)", use_container_width=True, key="d_heavy"):
            write_control({"drop_prob": 0.30})
        if st.button("Severe (50%)", use_container_width=True, key="d_severe"):
            write_control({"drop_prob": 0.50})
        
        st.divider()
        st.subheader("System Mode")
        if st.button("Raw", help="No protection", use_container_width=True, key="m_raw"):
            write_control({
                "recovery_enabled": False,
                "kalman_enabled": False,
                "predict_enabled": False,
            })
        if st.button("Basic", help="ZoH only", use_container_width=True, key="m_basic"):
            write_control({
                "recovery_enabled": True,
                "kalman_enabled": False,
                "predict_enabled": False,
            })
        if st.button("Full", help="Full recovery", use_container_width=True, key="m_full"):
            write_control({
                "recovery_enabled": True,
                "kalman_enabled": True,
                "predict_enabled": True,
            })
    
    # ----- Main Content -----
    st.title("ROI Privacy-Preserving Edge Streaming")
    st.caption("📺 Live video shown in separate OpenCV window")
    
    st_autorefresh(interval=REFRESH_MS, key="refresh")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    if "state_history" not in st.session_state:
        st.session_state.state_history = []
    
    stats = read_stats()
    
    if stats is None:
        st.warning("⏳ Waiting for system stats. Run the server-side stitcher.")
        st.stop()
    
    # Auto-reset history if system restarted
    current_frames = stats.get("frames_done", 0)
    if "last_frames_done" not in st.session_state:
        st.session_state.last_frames_done = 0

    if current_frames < st.session_state.last_frames_done:
        st.session_state.history = []
        st.session_state.state_history = []
    st.session_state.last_frames_done = current_frames

    # State Banner (smoothed)
    state = stats.get("state", "UNKNOWN")
    st.session_state.state_history.append(state)
    if len(st.session_state.state_history) > 5:
        st.session_state.state_history = st.session_state.state_history[-5:]
    
    state_smoothed = Counter(st.session_state.state_history).most_common(1)[0][0]
    emoji, color, label = STATE_COLORS.get(state_smoothed, STATE_COLORS["UNKNOWN"])
    
    st.markdown(
        f'<div class="state-banner" style="background-color:{color};">'
        f'{emoji} {label}'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    # Status row
    mode_emoji, mode_label = detect_mode(stats)
    drop_pct = f"{stats.get('drop_prob_current', 0):.0%}"
    quality = stats.get('quality', '--')
    tracks = stats.get('tracks_count', 0)
    
    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.markdown(f"**Mode**  \n{mode_emoji} {mode_label}")
    cs2.markdown(f"**Drop**  \n{drop_pct}")
    cs3.markdown(f"**Quality**  \n{quality}")
    cs4.markdown(f"**Tracks**  \n{tracks}")
    
    st.divider()
    
    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frames", stats.get("frames_done", 0))
    c2.metric("Recovered", stats.get("patches_recovered", 0))
    c3.metric("Predicted", stats.get("patches_predicted", 0))
    c4.metric("Failed", stats.get("patches_failed", 0))
    c5.metric("Ratio", f"{stats.get('current_ratio', 0):.2f}")
    
    # Chart
    snapshot = {
        "frame": stats.get("frames_done", 0),
        "ratio": stats.get("current_ratio", 0),
        "quality": stats.get("quality") or 0,
    }
    st.session_state.history.append(snapshot)
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]
    
    if len(st.session_state.history) > 2:
        df = pd.DataFrame(st.session_state.history)
        
        st.markdown("**📈 Reception Ratio**")
        st.line_chart(df.set_index("frame")[["ratio"]], use_container_width=True, height=150)

        st.markdown("**📈 Adaptive Quality**")
        st.line_chart(df.set_index("frame")[["quality"]], use_container_width=True, height=150)
    else:
        st.info("📈 Chart will appear after a few seconds of data...")

# ===== Tab 2: Comparison =====
with tab2:
    st.title("Side-by-Side Comparison")
    st.caption("Same video, same network conditions, different system modes.")
    
    # Stats summary
    st.subheader("Quantitative Results")
    summary_df = pd.DataFrame({
        "Scenario": ["Original", "Drop Only", "Full System"],
        "Drop Rate": ["0%", "30%", "30%"],
        "Patches Complete": [2190, 870, 894],
        "Recovered": [0, 0, 1146],
        "Predicted": [0, 0, 163],
        "NORMAL %": ["100%", "0.5%", "0.5%"],
        "EMERGENCY %": ["0%", "87.5%", "84.0%"],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    st.subheader("Visual Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Original")
        st.caption("No packet drop, baseline reference")
        video_path = COMPARISON_DIR / "comparison_original_h264.mp4"
        if video_path.exists():
            with open(video_path, "rb") as f:
                st.video(f.read())
        else:
            st.warning("Not found")
    
    with col2:
        st.markdown("##### Drop Only")
        st.caption("30% drop, no recovery")
        video_path = COMPARISON_DIR / "comparison_drop_only_h264.mp4"
        if video_path.exists():
            with open(video_path, "rb") as f:
                st.video(f.read())
        else:
            st.warning("Not found")
    
    with col3:
        st.markdown("##### Full System")
        st.caption("30% drop, ZoH + Kalman + Predict")
        video_path = COMPARISON_DIR / "comparison_full_h264.mp4"
        if video_path.exists():
            with open(video_path, "rb") as f:
                st.video(f.read())
        else:
            st.warning("Not found")
    
    st.divider()
    st.info(
        "**Key insight:** Under 30% packet loss, _Drop Only_ loses 1182 patches — "
        "people disappear from view. _Full System_ recovers 1146 patches via "
        "Zero-order Hold + Kalman, and generates 163 virtual patches via predict-only "
        "mode. Recovery preserves tracking continuity *and* slightly improves state "
        "stability (84% vs 87.5% Emergency)."
    )