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
    "NORMAL": ("🟢", "#28a745"),
    "WARNING": ("🟡", "#ffc107"),
    "EMERGENCY": ("🔴", "#dc3545"),
    "UNKNOWN": ("⚪", "#6c757d"),
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
        return "🔴 Raw"
    if not k and not p:
        return "🟡 Basic"
    if r and k and p:
        return "🟢 Full"
    return "Custom"


st.set_page_config(
    page_title="ROI Privacy Edge — Dashboard",
    layout="wide",
)

st.title("ROI Privacy-Preserving Edge Streaming")

# === Tabs ===
tab1, tab2 = st.tabs(["🔴 Live Demo", "📊 Comparison"])

# ===== Tab 1: Live Demo =====
with tab1:
    st.caption("Live video shown in separate OpenCV window. Use buttons below to adjust system in real-time.")
    
    st_autorefresh(interval=REFRESH_MS, key="dashboard_refresh")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    if "state_history" not in st.session_state:
        st.session_state.state_history = []
    
    stats = read_stats()
    
    if stats is None:
        st.info("⏳ Waiting for system stats. Run the server-side stitcher.")
    else:
        # State Banner (smoothed)
        state = stats.get("state", "UNKNOWN")
        st.session_state.state_history.append(state)
        if len(st.session_state.state_history) > 5:
            st.session_state.state_history = st.session_state.state_history[-5:]
        
        state_smoothed = Counter(st.session_state.state_history).most_common(1)[0][0]
        emoji, color = STATE_COLORS.get(state_smoothed, STATE_COLORS["UNKNOWN"])
        
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:1rem;
                        border-radius:0.5rem; text-align:center;
                        color:white; font-size:2rem; font-weight:bold;">
                {emoji} System State: {state_smoothed}
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Network Drop Buttons
        st.subheader("Network Drop")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        if col_d1.button("Clean (0%)"):
            write_control({"drop_prob": 0.0})
        if col_d2.button("Light (15%)"):
            write_control({"drop_prob": 0.15})
        if col_d3.button("Heavy (30%)"):
            write_control({"drop_prob": 0.30})
        if col_d4.button("Severe (50%)"):
            write_control({"drop_prob": 0.50})
        
        # System Mode Buttons
        st.subheader("System Mode")
        col_m1, col_m2, col_m3 = st.columns(3)
        if col_m1.button("Raw (No Protection)"):
            write_control({
                "recovery_enabled": False,
                "kalman_enabled": False,
                "predict_enabled": False,
            })
        if col_m2.button("Basic (ZoH only)"):
            write_control({
                "recovery_enabled": True,
                "kalman_enabled": False,
                "predict_enabled": False,
            })
        if col_m3.button("Full System ⭐"):
            write_control({
                "recovery_enabled": True,
                "kalman_enabled": True,
                "predict_enabled": True,
            })
        
        # Current Config
        st.markdown(
            f"**Current Mode:** {detect_mode(stats)} | "
            f"**Drop:** {stats.get('drop_prob_current', 0):.1%} | "
            f"**Quality:** {stats.get('quality', 'N/A')} | "
            f"**Tracks:** {stats.get('tracks_count', 0)}"
        )
        
        # Metric Cards
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
            st.subheader("Reception Ratio & Quality")
            df = pd.DataFrame(st.session_state.history)
            st.line_chart(df.set_index("frame")[["ratio", "quality"]])

# ===== Tab 2: Comparison =====
with tab2:
    st.header("Side-by-Side Comparison")
    st.caption("Same video, same network conditions, different system modes. Demonstrates system's recovery capability.")
    
    # Stats summary table
    st.subheader("Quantitative Results")
    summary_df = pd.DataFrame({
        "Scenario": ["Original", "Drop Only", "Full System"],
        "Drop Rate": ["0%", "30%", "30%"],
        "Patches Complete": [2168, 894, 890],
        "Patches Recovered": [1, 0, 1135],
        "Patches Predicted": [74, 0, 192],
        "State NORMAL": ["100%", "0.5%", "0.5%"],
        "State EMERGENCY": ["0%", "84%", "89%"],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    st.subheader("Visual Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🟢 Original")
        st.caption("Baseline: no packet drop, all protection")
        video_path = COMPARISON_DIR / "comparison_original_h264.mp4"
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.warning(f"Video not found: {video_path}")
    
    with col2:
        st.markdown("### 🔴 Drop Only")
        st.caption("30% packet drop, no recovery")
        video_path = COMPARISON_DIR / "comparison_drop_only_h264.mp4"
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.warning(f"Video not found: {video_path}")
    
    with col3:
        st.markdown("### 🟢 Full System")
        st.caption("30% packet drop, with ZoH + Kalman + Predict recovery")
        video_path = COMPARISON_DIR / "comparison_full_h264.mp4"
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.warning(f"Video not found: {video_path}")
    
    st.divider()
    st.info(
        "**Key finding:** Under 30% packet loss, Drop Only loses 57% of patches (people disappear). "
        "Full System recovers 1135 patches via Zero-order Hold + Kalman, and generates 192 virtual "
        "patches via predict-only mode, maintaining continuous tracking."
    )