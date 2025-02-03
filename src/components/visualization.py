import streamlit as st
import plotly.graph_objects as go

def render_report(report: dict):
    """Render the engagement analysis report"""
    st.header("Engagement Analysis Report")
    
    # Overall Score
    score_color = _get_score_color(report['overall_score'])
    st.metric(
        label="Overall Class Engagement Score",
        value=f"{report['overall_score']:.2f}",
        delta=None,
    )
    
    # Student Timeline
    st.subheader("Individual Student Engagement")
    st.plotly_chart(report['student_timeline'], use_container_width=True)
    
    # Class Timeline
    st.subheader("Class Average Engagement")
    st.plotly_chart(report['class_timeline'], use_container_width=True)
    
    # Engagement Distribution
    st.subheader("Engagement Level Distribution")
    st.plotly_chart(report['engagement_distribution'], use_container_width=True)

def _get_score_color(score: float) -> str:
    """Get color based on engagement score"""
    if score >= 0.7:
        return "green"
    elif score >= 0.4:
        return "yellow"
    return "red"
