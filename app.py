import streamlit as st
import time
import sys
import os
import pandas as pd
from datetime import datetime

# Ensure we can import from backend
# sys.path.append(os.path.join(os.path.dirname(__file__), 'backend')) # REMOVED

from core.tracker import BackgroundTracker
from core.features import FeatureExtractor
from core import config

# Page Config
st.set_page_config(
    page_title="StressTracker AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Deep Space" Aesthetic
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at center, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(233, 69, 96, 0.6);
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 200;
        letter-spacing: 0.1em;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-val {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(#e94560, #fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'tracker' not in st.session_state:
    st.session_state.tracker = BackgroundTracker()
if 'is_tracking' not in st.session_state:
    st.session_state.is_tracking = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ CONFIGURATION")
    duration = st.select_slider(
        "Session Duration",
        options=[config.SESSION_DURATION, 180, 300],
        value=config.SESSION_DURATION,
        format_func=lambda x: f"{x//60} min"
    )
    
    st.markdown("---")
    st.info("""
    **How to use:**
    1. Click START.
    2. Work normally in any app.
    3. The app tracks mouse/keyboard in background.
    4. Auto-finishes when timer ends.
    """)

    st.markdown("### 🔬 MODE")
    mode = st.radio("Select Mode", ["Live Analysis", "Calibration"], index=0)
    
    if mode == "Calibration":
        st.warning(f"Calibration establishes your 'Baseline'. Perform typical tasks (typing + mousing) for {config.SESSION_DURATION} seconds.")
        duration = config.SESSION_DURATION

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.title("STRESS TRACKER")
    st.markdown("##### AI-POWERED BIOMETRIC ANALYSIS")

    if not st.session_state.is_tracking:
        if st.button("START SESSION"):
            try:
                st.session_state.is_tracking = True
                st.session_state.start_time = time.time()
                st.session_state.analysis_result = None
                st.session_state.tracker.start()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start tracker: {e}")
                st.session_state.is_tracking = False
    else:
        # Tracking Phase
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, duration - elapsed)
        
        # Countdown Display
        st.markdown(f"""
        <div style="font-size: 6rem; font-weight: 100; text-align: center; font-variant-numeric: tabular-nums;">
            {int(remaining // 60):02d}:{int(remaining % 60):02d}
        </div>
        <div style="text-align: center; letter-spacing: 0.5em; opacity: 0.7; margin-bottom: 2rem;">
            MONITORING SYSTEM ACTIVE
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-finish logic
        if remaining <= 0:
            st.session_state.is_tracking = False
            # Determine endpoint
            is_calib = (mode == "Calibration")
            
            try:
                # Stop Tracker
                session_data = st.session_state.tracker.stop()
                
                # Add config to data
                session_data['analyze_with_llm'] = not is_calib # No LLM for calibration, just stats
                
                # Send to Backend
                from core.features import FeatureExtractor
                import core.analysis as analysis
                import core.utils as utils
                
                # Extract basic features for local display
                key_feats = FeatureExtractor.extract_keystroke_features(session_data['keystrokes'])
                mouse_feats = FeatureExtractor.extract_mouse_features(session_data['movements'])
                
                if is_calib:
                    # Call Calibrate
                    with st.spinner("Calibrating Baseline..."):
                        s_data = analysis.SessionData(**session_data)
                        res = analysis.calibrate_session_logic(s_data)
                        st.success("✅ Baseline Established! You can now switch to 'Live Analysis'.")
                        st.json(res['baseline'])
                else:
                    # Normal Analysis
                    with st.spinner("Analyzing Biometrics with Z-Scoring & Agent..."):
                        s_data = analysis.SessionData(**session_data)
                        res = analysis.submit_session_logic(s_data)
                        
                        st.session_state.analysis_result = {
                            "score": res['stress_score'],
                            "features": res['features'],
                            "z_scores": res.get('z_scores', {}),
                            "analysis": res['llm_analysis']
                        }
                        st.rerun()
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                # Ideally log this too
        else:
            time.sleep(0.1)
            st.rerun()

    # Results Display
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        score_percent = int(res['score'] * 100)
        
        # Score Card
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1rem; text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 0.5rem;">Stress Score</div>
            <div class="metric-val" style="color: {'#34d399' if score_percent < 30 else '#fbbf24' if score_percent < 70 else '#f43f5e'}">
                {score_percent}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🩺 Clinical Assessment")
        st.write(res['analysis'].get('clinical_assessment', 'No assessment available.'))
        
        st.markdown("#### ⚠️ Symptom Clusters")
        for sym in res['analysis'].get('symptom_clusters', []):
            st.warning(sym, icon="⚠️")

        st.markdown("#### ⚡ Immediate Action")
        st.success(res['analysis'].get('immediate_action', 'No immediate action required.'), icon="⚡")

        st.markdown("#### 📋 Recommendations")
        for rec in res['analysis'].get('recommendations', []):
            with st.expander(f"**{rec.get('title', 'Recommendation')}**"):
                st.write(rec.get('description', ''))

        # --- Data Export ---
        st.markdown("---")
        st.subheader("📁 Research Export")
        
        # Create DataFrames
        mouse_df = pd.DataFrame(st.session_state.tracker.mouse_data)
        key_df = pd.DataFrame(st.session_state.tracker.keystrokes)
        
        # Convert to CSV
        csv_mouse = mouse_df.to_csv(index=False).encode('utf-8')
        csv_key = key_df.to_csv(index=False).encode('utf-8')
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 Download Mouse Data (CSV)",
                csv_mouse,
                "mouse_data.csv",
                "text/csv",
                key='download-mouse'
            )
        with c2:
            st.download_button(
                "📥 Download Keystroke Data (CSV)",
                csv_key,
                "keystroke_data.csv",
                "text/csv",
                key='download-key'
            )

        # --- Chat Interface ---
        st.markdown("---")
        st.subheader("💬 Chat with Dr. AI")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        if prompt := st.chat_input("Ask about your stress analysis..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.spinner("Thinking..."):
                from core.agent import StressManagementAgent
                # We need to re-init agent or use a persistent one (st.cache_resource would be better)
                if 'agent_instance' not in st.session_state:
                     st.session_state.agent_instance = StressManagementAgent(model_name=config.OLLAMA_MODEL)
                
                # Context is the analysis text
                context_str = f"Assessment: {res['analysis'].get('clinical_assessment')}. " \
                              f"Stress Score: {score_percent}%. " \
                              f"Recommendations: {str(res['analysis'].get('recommendations'))}"
                
                response = st.session_state.agent_instance.chat_response(prompt, context_str)
                
            # Add assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

with col2:
    # Live/Static Metrics
    st.markdown("### 📊 Metrics")
    
    if st.session_state.analysis_result:
        feats = st.session_state.analysis_result['features']
        z_scores = st.session_state.analysis_result.get('z_scores', {})
        
        def display_z_metric(label, value, z_key, citation):
            z = z_scores.get(z_key, 0.0)
            delta_color = "normal"
            if z > 1.5: delta_color = "inverse" # High stress?
            elif z < -1.5: delta_color = "off"
            
            st.metric(
                label=label,
                value=f"{value:.2f}",
                delta=f"{z:+.1f} σ (Z-Score)",
                delta_color=delta_color
            )
            st.caption(f"*{citation}*")
            st.markdown("---")

        display_z_metric(
            "Mouse Jitter (Acc Std)", 
            feats.get('mouse_acc_std', 0), 
            'z_mouse_acc_std',
            "Bioindicator: Sympathetic Activation"
        )
        
        display_z_metric(
            "Flight Time Var (ms)", 
            feats.get('key_flight_std', 0), 
            'z_key_flight_std',
            "Source: Epp et al. (2011) - Cognitive Load"
        )
        
        display_z_metric(
            "Path Efficiency (1.0=Ideal)", 
            feats.get('mouse_path_efficiency', 1.0), 
            'z_mouse_path_efficiency',
            "Source: Movement Efficiency Index"
        )
        
        display_z_metric(
            "Click Latency (ms)", 
            feats.get('mouse_click_latency', 0), 
            'z_mouse_click_latency',
            "Indicator: Psychomotor Hesitation"
        )
        
    elif st.session_state.is_tracking:
        st.info("Collecting data...")
        st.metric("Mouse Events", len(st.session_state.tracker.movements))
        st.metric("Keystrokes", len(st.session_state.tracker.keystrokes))

    # --- References Section ---
    st.markdown("---")
    st.markdown("### 📚 Clinical References")
    st.info("""
    **Digital Phenotyping:**
    *   *Insel, T. R. (2017).* Digital Phenotyping: Technology for a New Science of Behavior. JAMA.
    *   *Epp, C., et al. (2011).* Identifying emotional states using keystroke dynamics.
    
    **Analysis Methods:**
    *   **Keystroke Dynamics**: Rhythm variance as a proxy for cognitive load (Yerkes-Dodson Law).
    *   **Kinematic Analysis**: Mouse jitter correlates with sympathetic nervous system activation.
    """)
