# src/data/validator.py
import pandas as pd
import numpy as np
from src.utils.logger import logger
from src.config import REQUIRED_COLUMNS, TARGET_COLUMN

class DataValidator:
    """Validate dataset structure and content"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataset(self, data):
        """Validate the dataset"""
        logger.info("Validating dataset...")
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Check if data is not empty
        if data is None or len(data) == 0:
            results['valid'] = False
            results['errors'].append("Dataset is empty")
            return results
        
        # Check required columns - make it flexible
        required_for_validation = [
            'historical_grade', 'previous_grade', 'assignment_score',
            'quiz_score', 'midterm_score', 'attendance_percentage',
            'lms_login_count', 'lms_activity_count', 'study_hours'
        ]
        
        missing_cols = [col for col in required_for_validation if col not in data.columns]
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")
            return results
        
        # Check target column
        if TARGET_COLUMN not in data.columns:
            results['valid'] = False
            results['errors'].append(f"Target column '{TARGET_COLUMN}' not found")
            return results
        
        # Check target values - accept PASS/FAIL or SAFE/AT_RISK/FAIL
        target_values = data[TARGET_COLUMN].unique()
        valid_targets = ['PASS', 'FAIL', 'SAFE', 'AT_RISK']
        invalid_targets = [v for v in target_values if v not in valid_targets]
        if invalid_targets:
            results['warnings'].append(f"Unusual target values found: {invalid_targets}")
            # Still valid, just warning
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            missing_dict = missing[missing > 0].to_dict()
            results['warnings'].append(f"Missing values found in columns: {missing_dict}")
            results['info']['missing_values'] = missing_dict
            
            # If too many missing values, mark as invalid
            for col, count in missing_dict.items():
                if count / len(data) > 0.5:  # More than 50% missing
                    results['valid'] = False
                    results['errors'].append(f"Column '{col}' has {count} missing values ({count/len(data)*100:.1f}%)")
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            results['warnings'].append(f"Found {duplicates} duplicate rows")
            results['info']['duplicates'] = duplicates
        
        # Check data types - ensure numerical columns are numeric
        numeric_columns = ['historical_grade', 'previous_grade', 'assignment_score', 
                          'quiz_score', 'midterm_score', 'attendance_percentage',
                          'lms_login_count', 'lms_activity_count', 'study_hours', 'age']
        
        for col in numeric_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        results['warnings'].append(f"Converted '{col}' to numeric")
                    except:
                        results['errors'].append(f"Column '{col}' should be numeric")
                        results['valid'] = False
        
        # Check value ranges
        range_checks = {
            'historical_grade': (0, 100),
            'previous_grade': (0, 100),
            'assignment_score': (0, 100),
            'quiz_score': (0, 100),
            'midterm_score': (0, 100),
            'attendance_percentage': (0, 100),
            'study_hours': (0, 40),
            'age': (16, 60)
        }
        
        for col, (min_val, max_val) in range_checks.items():
            if col in data.columns:
                out_of_range = ((data[col] < min_val) | (data[col] > max_val)).sum()
                if out_of_range > 0:
                    results['warnings'].append(f"Column '{col}' has {out_of_range} values outside range [{min_val}, {max_val}]")
        
        # Check for outliers in numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'student_id':
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers > 0 and outliers < len(data) * 0.1:  # Less than 10% outliers
                    results['warnings'].append(f"Column '{col}' has {outliers} potential outliers")
        
        # Check class balance
        if TARGET_COLUMN in data.columns:
            class_counts = data[TARGET_COLUMN].value_counts()
            if len(class_counts) < 2:
                results['warnings'].append(f"Only one class found in target: {class_counts.index[0]}")
                results['valid'] = False
                results['errors'].append("Target column must have at least two classes (PASS/FAIL)")
        
        self.validation_results = results
        logger.info(f"Validation complete. Valid: {results['valid']}")
        return results
    
    def get_validation_summary(self):
        """Get validation summary"""
        return self.validation_results