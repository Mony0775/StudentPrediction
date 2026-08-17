# src/features/engineering.py
import pandas as pd
import numpy as np
from src.utils.logger import logger

class FeatureEngineer:
    """Create derived features from existing data"""
    
    def __init__(self):
        self.engineered_features = []
        
    def engineer_features(self, data):
        """Create new features from the dataset"""
        logger.info("Engineering new features...")
        df = data.copy()
        
        # Average assessment score
        if all(col in df.columns for col in ['assignment_score', 'quiz_score', 'midterm_score']):
            df['average_assessment_score'] = df[['assignment_score', 'quiz_score', 'midterm_score']].mean(axis=1)
            self.engineered_features.append('average_assessment_score')
        
        # Attendance risk
        if 'attendance_percentage' in df.columns:
            df['attendance_risk'] = df['attendance_percentage'].apply(
                lambda x: 1 if x < 70 else 0
            )
            self.engineered_features.append('attendance_risk')
        
        # Low grade flag
        if 'historical_grade' in df.columns:
            df['low_grade_flag'] = df['historical_grade'].apply(
                lambda x: 1 if x < 50 else 0
            )
            self.engineered_features.append('low_grade_flag')
        
        # Engagement score
        if all(col in df.columns for col in ['lms_login_count', 'lms_activity_count']):
            # Normalize and combine
            login_norm = df['lms_login_count'] / df['lms_login_count'].max() if df['lms_login_count'].max() > 0 else 0
            activity_norm = df['lms_activity_count'] / df['lms_activity_count'].max() if df['lms_activity_count'].max() > 0 else 0
            df['engagement_score'] = (login_norm * 0.4 + activity_norm * 0.6) * 100
            self.engineered_features.append('engagement_score')
        
        # Academic progress score
        if all(col in df.columns for col in ['historical_grade', 'previous_grade']):
            df['academic_progress_score'] = df['historical_grade'] - df['previous_grade']
            self.engineered_features.append('academic_progress_score')
        
        # Total LMS engagement
        if all(col in df.columns for col in ['lms_login_count', 'lms_activity_count']):
            df['total_lms_engagement'] = df['lms_login_count'] + df['lms_activity_count']
            self.engineered_features.append('total_lms_engagement')
        
        # Performance index (combination of key metrics)
        metrics = ['historical_grade', 'previous_grade', 'assignment_score', 'quiz_score', 'midterm_score']
        available_metrics = [m for m in metrics if m in df.columns]
        if available_metrics:
            df['performance_index'] = df[available_metrics].mean(axis=1)
            self.engineered_features.append('performance_index')
        
        logger.info(f"Engineered {len(self.engineered_features)} new features: {self.engineered_features}")
        return df
    
    def get_engineered_feature_names(self):
        """Get list of engineered feature names"""
        return self.engineered_features