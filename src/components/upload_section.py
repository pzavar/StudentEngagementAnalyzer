import streamlit as st

def render_upload_section():
    """Render the video upload section"""
    st.title("Student Engagement Analysis")
    st.write("""
    Upload a recorded Zoom lecture video to analyze student engagement levels.
    The system will process the video and generate an engagement report.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov'],
        help="Upload a recorded Zoom lecture video"
    )
    
    return uploaded_file
