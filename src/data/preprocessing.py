# src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils.logger import logger
from src.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN

class DataPreprocessor:
    """Handle data preprocessing"""
    
    def __init__(self):
        self.preprocessor = None
        self.label_encoder = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def preprocess(self, data, target_column=TARGET_COLUMN):
        """Preprocess the dataset"""
        logger.info("Starting preprocessing...")
        
        # Make a copy
        df = data.copy()
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_len - len(df)} duplicate rows")
        
        # Separate features and target
        if target_column in df.columns:
            y = df[target_column].copy()
            X = df.drop(columns=[target_column])
            
            # Encode target
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.y = y_encoded
        else:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Remove student_id from features
        if 'student_id' in X.columns:
            X = X.drop(columns=['student_id'])
        
        # Identify numerical and categorical columns
        # Numerical columns are those that are numeric and in NUMERICAL_COLUMNS
        numerical_cols = []
        for col in NUMERICAL_COLUMNS:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                numerical_cols.append(col)
        
        # Categorical columns are those that are object type or in CATEGORICAL_COLUMNS
        categorical_cols = []
        for col in CATEGORICAL_COLUMNS:
            if col in X.columns:
                categorical_cols.append(col)
        
        # Also check for any other object type columns
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            if col not in categorical_cols and col != 'student_id':
                categorical_cols.append(col)
        
        # Ensure numerical columns are not in categorical
        categorical_cols = [col for col in categorical_cols if col not in numerical_cols]
        
        logger.info(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        logger.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        # Fit and transform
        X_processed = preprocessor.fit_transform(X)
        
        # Build feature names with clean labels
        feature_names = []
        
        # Define clean name mapping
        name_mapping = {
            'historical_grade': 'Historical Grade',
            'previous_grade': 'Previous Grade',
            'assignment_score': 'Assignment Score',
            'quiz_score': 'Quiz Score',
            'midterm_score': 'Midterm Score',
            'attendance_percentage': 'Attendance Percentage',
            'lms_login_count': 'LMS Login Count',
            'lms_activity_count': 'LMS Activity Count',
            'study_hours': 'Study Hours',
            'age': 'Age',
            'gender': 'Gender',
            'department': 'Department',
            'course': 'Course'
        }
        
        # Numerical features - use clean names
        for col in numerical_cols:
            clean_name = name_mapping.get(col, col.replace('_', ' ').title())
            feature_names.append(clean_name)
        
        # Categorical features - create descriptive names
        for i, col in enumerate(categorical_cols):
            categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i]
            col_clean = name_mapping.get(col, col.title())
            for cat in categories:
                clean_name = f"{col_clean}: {cat}"
                feature_names.append(clean_name)
        
        # Remove any duplicates while preserving order
        seen = set()
        unique_feature_names = []
        for name in feature_names:
            if name not in seen:
                seen.add(name)
                unique_feature_names.append(name)
        
        self.feature_names = unique_feature_names
        
        self.X = X_processed
        self.preprocessor = preprocessor
        
        logger.info(f"Preprocessing complete. Features: {self.X.shape[1]}, Samples: {self.X.shape[0]}")
        logger.info(f"Feature names ({len(self.feature_names)}): {self.feature_names}")
        return self.X, self.y
    
    def transform(self, data):
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call preprocess first.")
        
        df = data.copy()
        
        # Remove student_id if present
        if 'student_id' in df.columns:
            df = df.drop(columns=['student_id'])
        
        # Remove target if present
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])
        
        return self.preprocessor.transform(df)
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        return self.feature_names