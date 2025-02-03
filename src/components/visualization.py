"""
Visualization component for the Student Engagement Analysis application.
Handles all data visualization and report generation using Plotly.
"""
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any
import plotly.express as px

def render_report(report: Dict[str, Any]) -> None:
    """
    Render the engagement analysis report with interactive visualizations.

    Args:
        report (Dict[str, Any]): Dictionary containing report data including:
            - overall_score (float): Overall class engagement score
            - student_timeline (plotly.Figure): Individual student engagement over time
            - class_timeline (plotly.Figure): Class average engagement over time
            - engagement_distribution (plotly.Figure): Distribution of engagement levels
    """
    st.header("Engagement Analysis Report")

    # Overall Score with custom styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        score_color = _get_score_color(report['overall_score'])
        st.metric(
            label="Overall Class Engagement Score",
            value=f"{report['overall_score']:.2f}",
            delta=None,
            help="Average engagement score across all students throughout the lecture"
        )

    # Student Timeline with enhanced interactivity
    st.subheader("Individual Student Engagement")
    st.plotly_chart(
        report['student_timeline'],
        use_container_width=True,
        config={'displayModeBar': True}
    )

    # Class Timeline with tooltips
    st.subheader("Class Average Engagement")
    st.plotly_chart(
        report['class_timeline'],
        use_container_width=True,
        config={'displayModeBar': True}
    )

    # Engagement Distribution with legend
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Engagement Level Distribution")
        st.plotly_chart(
            report['engagement_distribution'],
            use_container_width=True,
            config={'displayModeBar': True}
        )
    with col2:
        st.markdown("""
        ### Understanding the Metrics
        - **Highly Engaged**: Active participation and positive emotions
        - **Moderately Engaged**: Neutral attention levels
        - **Not Engaged**: Signs of disinterest or negative emotions
        """)

def _get_score_color(score: float) -> str:
    """
    Get color based on engagement score for visual feedback.

    Args:
        score (float): Engagement score between 0 and 1

    Returns:
        str: Color code corresponding to the score level
    """
    if score >= 0.7:
        return "#28a745"  # Green
    elif score >= 0.4:
        return "#ffc107"  # Yellow
    return "#dc3545"  # Red