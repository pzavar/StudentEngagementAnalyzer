import streamlit as st

def render_sidebar():
    """Render the sidebar with configuration options"""
    st.sidebar.title("Configuration")
    
    sampling_rate = st.sidebar.slider(
        "Sampling Rate (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        step=30,
        help="Time interval between frame captures"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Minimum confidence score for emotion detection"
    )
    
    return {
        "sampling_rate": sampling_rate,
        "confidence_threshold": confidence_threshold
    }
