import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List

class ReportGenerator:
    def __init__(self):
        self.engagement_scores = {
            'highly engaged': 1.0,
            'moderately engaged': 0.5,
            'not engaged': 0.0
        }

    def generate_report(self, engagement_data: List[Dict]) -> Dict:
        """Generate engagement report from processed data"""
        df = pd.DataFrame(engagement_data)
        
        # Overall engagement score
        df['engagement_score'] = df['engagement_level'].map(self.engagement_scores)
        overall_score = df['engagement_score'].mean()
        
        # Individual student engagement over time
        student_timeline = self._create_student_timeline(df)
        
        # Class engagement trends
        class_timeline = self._create_class_timeline(df)
        
        # Engagement distribution
        engagement_dist = self._create_engagement_distribution(df)
        
        return {
            'overall_score': overall_score,
            'student_timeline': student_timeline,
            'class_timeline': class_timeline,
            'engagement_distribution': engagement_dist
        }

    def _create_student_timeline(self, df: pd.DataFrame) -> go.Figure:
        """Create timeline of engagement for each student"""
        fig = px.line(df, x='timestamp', y='engagement_score', 
                     color='student_id', title='Student Engagement Timeline')
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Engagement Score",
            template="plotly_white"
        )
        return fig

    def _create_class_timeline(self, df: pd.DataFrame) -> go.Figure:
        """Create timeline of average class engagement"""
        class_avg = df.groupby('timestamp')['engagement_score'].mean().reset_index()
        fig = px.line(class_avg, x='timestamp', y='engagement_score',
                      title='Class Average Engagement')
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Average Engagement Score",
            template="plotly_white"
        )
        return fig

    def _create_engagement_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create distribution of engagement levels"""
        dist = df['engagement_level'].value_counts()
        fig = px.pie(values=dist.values, names=dist.index,
                    title='Distribution of Engagement Levels')
        fig.update_layout(template="plotly_white")
        return fig
