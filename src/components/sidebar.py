"""
Sidebar component for the Student Engagement Analysis application.
Handles configuration settings and analysis parameters.
"""
import streamlit as st
from typing import Dict, Any

def render_sidebar() -> Dict[str, Any]:
    """
    Render the configuration sidebar with analysis parameters.

    Returns:
        Dict[str, Any]: Configuration parameters including:
            - sampling_rate (int): Time interval between frame captures
            - confidence_threshold (float): Minimum confidence score for emotion detection
    """
    with st.sidebar:
        st.title("Analysis Configuration")

        st.markdown("""
        ### Settings
        Adjust these parameters to fine-tune the analysis process.
        """)

        # Sampling rate with detailed help text
        sampling_rate = st.slider(
            "Sampling Rate (seconds)",
            min_value=30,
            max_value=300,
            value=60,
            step=30,
            help="Time interval between video frame captures. Lower values provide more "
                 "detailed analysis but require more processing time."
        )

        # Confidence threshold with explanation
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence level required for emotion detection. Higher values "
                 "ensure more reliable predictions but may reduce the number of detections."
        )

        st.markdown("""
        ### Tips
        - For longer videos, consider increasing the sampling rate
        - Adjust confidence threshold based on lighting conditions
        """)

        return {
            "sampling_rate": sampling_rate,
            "confidence_threshold": confidence_threshold
        }