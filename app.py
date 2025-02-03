import streamlit as st
from src.components.sidebar import render_sidebar
from src.components.upload_section import render_upload_section
from src.components.visualization import render_report
from src.models.emotion_classifier import EmotionClassifier
from src.utils.video_processor import VideoProcessor
from src.utils.face_extractor import FaceExtractor
from src.utils.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Student Engagement Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()
if 'face_extractor' not in st.session_state:
    st.session_state.face_extractor = FaceExtractor()
if 'classifier' not in st.session_state:
    st.session_state.classifier = EmotionClassifier()
if 'report_generator' not in st.session_state:
    st.session_state.report_generator = ReportGenerator()

def process_video(video_file, config):
    """Process video and generate engagement data"""
    engagement_data = []
    
    with st.spinner('Processing video...'):
        progress_bar = st.progress(0)
        
        for timestamp, frame in st.session_state.processor.process_video(video_file):
            faces = st.session_state.face_extractor.extract_faces(frame)
            
            for student_id, face_image in faces:
                emotion, confidence = st.session_state.classifier.predict_emotion(face_image)
                
                if confidence >= config['confidence_threshold']:
                    engagement_level = st.session_state.classifier.map_to_engagement(emotion)
                    engagement_data.append({
                        'timestamp': timestamp,
                        'student_id': student_id,
                        'emotion': emotion,
                        'engagement_level': engagement_level
                    })
            
            progress_bar.progress(min(timestamp / video_file.size, 1.0))
    
    return engagement_data

def main():
    # Render sidebar
    config = render_sidebar()
    
    # Render upload section
    uploaded_file = render_upload_section()
    
    if uploaded_file is not None:
        # Process video
        engagement_data = process_video(uploaded_file, config)
        
        if engagement_data:
            # Generate and render report
            report = st.session_state.report_generator.generate_report(engagement_data)
            render_report(report)
        else:
            st.warning("No faces detected in the video.")

if __name__ == "__main__":
    main()
