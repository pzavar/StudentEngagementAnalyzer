import streamlit as st
import os

def render_upload_section():
    """Render the video upload section"""
    st.title("Student Engagement Analysis")
    st.write("""
    Upload a recorded Zoom lecture video to analyze student engagement levels.
    The system will process the video and generate an engagement report.
    """)

    # Check for model file
    if not os.path.exists("model.tflite"):
        st.warning("""
        ⚠️ Model file not found. Before using this application:
        1. Train the emotion classification model using the provided training script
        2. Copy the generated 'model.tflite' file to the project root directory

        See README.md for detailed instructions on model training.
        """)

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov'],
        help="Upload a recorded Zoom lecture video"
    )

    return uploaded_file