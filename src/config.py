# src/config.py
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
STATIC_DIR = BASE_DIR / 'static'
TEMPLATES_DIR = BASE_DIR / 'templates'

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset settings
DEFAULT_DATASET_PATH = RAW_DATA_DIR / 'student_performance.csv'
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / 'processed_data.csv'

# Model settings - Using .pkl extension
BEST_MODEL_PATH = MODELS_DIR / 'best_model.pkl'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.pkl'
FEATURE_SELECTOR_PATH = MODELS_DIR / 'feature_selector.pkl'
LABEL_ENCODER_PATH = MODELS_DIR / 'label_encoder.pkl'
METRICS_PATH = MODELS_DIR / 'model_metrics.json'

# Individual model paths
LOGISTIC_REGRESSION_PATH = MODELS_DIR / 'logistic_regression.pkl'
DECISION_TREE_PATH = MODELS_DIR / 'decision_tree.pkl'
RANDOM_FOREST_PATH = MODELS_DIR / 'random_forest.pkl'
NAIVE_BAYES_PATH = MODELS_DIR / 'naive_bayes.pkl'

# ML settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N_FEATURES = 10

# Risk thresholds - probability of FAILURE
RISK_THRESHOLDS = {
    'LOW': 0.30,      # 0-30% failure probability = LOW RISK
    'MEDIUM': 0.60,   # 31-60% failure probability = MEDIUM RISK
    'HIGH': 0.80,     # 61-80% failure probability = HIGH RISK
    'CRITICAL': 1.00  # 81-100% failure probability = CRITICAL RISK
}

# Target column
TARGET_COLUMN = 'performance_status'

# Feature columns (required for validation)
REQUIRED_COLUMNS = [
    'student_id', 'gender', 'age', 'department', 'course',
    'historical_grade', 'previous_grade', 'assignment_score',
    'quiz_score', 'midterm_score', 'attendance_percentage',
    'lms_login_count', 'lms_activity_count', 'study_hours'
]

# Categorical columns - these will be one-hot encoded
CATEGORICAL_COLUMNS = ['gender', 'department', 'course']

# Numerical columns - these will be scaled
NUMERICAL_COLUMNS = [
    'age', 'historical_grade', 'previous_grade', 'assignment_score',
    'quiz_score', 'midterm_score', 'attendance_percentage',
    'lms_login_count', 'lms_activity_count', 'study_hours'
]

# Risk level recommendations
RECOMMENDATIONS = {
    'LOW': 'Student performance appears stable. Continue regular monitoring.',
    'MEDIUM': 'Monitor academic progress and encourage increased engagement.',
    'HIGH': 'Academic support is recommended. Review attendance and assessment performance.',
    'CRITICAL': 'Immediate intervention is recommended. Contact the student and provide academic support.'
}