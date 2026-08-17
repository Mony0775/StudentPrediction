# src/data/loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import logger
from src.config import RAW_DATA_DIR, REQUIRED_COLUMNS

class DataLoader:
    """Handle data loading operations"""
    
    def __init__(self):
        self.data = None
        self.file_path = None
    
    def load_csv(self, file_path):
        """Load CSV file"""
        try:
            self.file_path = Path(file_path)
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.data)} rows from {file_path}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_synthetic_data(self):
        """Generate synthetic student data with realistic patterns"""
        logger.info("Generating synthetic dataset...")
        
        np.random.seed(42)
        n_students = 5000
        
        # Generate student IDs
        student_ids = [f"S{str(i).zfill(4)}" for i in range(1, n_students + 1)]
        
        # Generate demographic data
        genders = np.random.choice(['Male', 'Female'], n_students, p=[0.52, 0.48])
        ages = np.random.randint(18, 30, n_students)
        departments = np.random.choice(
            ['Computer Science', 'Engineering', 'Business', 'Mathematics', 'Physics'],
            n_students,
            p=[0.30, 0.25, 0.20, 0.15, 0.10]
        )
        courses = np.random.choice(
            ['CS101', 'CS201', 'ENG101', 'BUS101', 'MATH101', 'PHY101'],
            n_students
        )
        
        # Generate academic data with realistic relationships
        # Base academic ability (influences everything)
        base_ability = np.random.normal(0, 1, n_students)
        
        # Historical grades (influenced by ability)
        historical_grades = np.clip(70 + base_ability * 15 + np.random.normal(0, 8, n_students), 0, 100)
        
        # Previous grades (correlated with historical)
        previous_grades = np.clip(historical_grades + np.random.normal(0, 5, n_students), 0, 100)
        
        # Assignment scores
        assignment_scores = np.clip(65 + base_ability * 12 + np.random.normal(0, 10, n_students), 0, 100)
        
        # Quiz scores
        quiz_scores = np.clip(60 + base_ability * 10 + np.random.normal(0, 12, n_students), 0, 100)
        
        # Midterm scores
        midterm_scores = np.clip(historical_grades * 0.6 + np.random.normal(0, 12, n_students), 0, 100)
        
        # Attendance (lower for lower-performing students)
        attendance_base = 90 - (100 - historical_grades) * 0.3
        attendance = np.clip(attendance_base + np.random.normal(0, 10, n_students), 0, 100)
        
        # LMS activity (correlated with performance)
        lms_login_base = 20 + (historical_grades / 100) * 30
        lms_login = np.clip(lms_login_base + np.random.normal(0, 8, n_students), 0, 60)
        
        lms_activity_base = 50 + (historical_grades / 100) * 50
        lms_activity = np.clip(lms_activity_base + np.random.normal(0, 15, n_students), 0, 150)
        
        # Study hours (correlated with performance)
        study_hours = np.clip(5 + (historical_grades / 100) * 15 + np.random.normal(0, 3, n_students), 0, 30)
        
        # Determine performance status based on multiple factors
        performance_score = (
            historical_grades * 0.20 +
            previous_grades * 0.15 +
            assignment_scores * 0.20 +
            quiz_scores * 0.15 +
            midterm_scores * 0.15 +
            attendance * 0.10 +
            (lms_login / 60) * 100 * 0.03 +
            (lms_activity / 150) * 100 * 0.02
        )
        
        # Add some noise
        performance_score += np.random.normal(0, 5, n_students)
        
        # Classify into PASS/FAIL (threshold at 50)
        performance_status = np.where(performance_score >= 50, 'PASS', 'FAIL')
        
        # Ensure we have both PASS and FAIL
        if np.sum(performance_status == 'PASS') < 100:
            # Force some PASSes
            indices = np.random.choice(n_students, 200, replace=False)
            performance_status[indices] = 'PASS'
        
        # Create DataFrame
        data = pd.DataFrame({
            'student_id': student_ids,
            'gender': genders,
            'age': ages,
            'department': departments,
            'course': courses,
            'historical_grade': np.round(historical_grades, 1),
            'previous_grade': np.round(previous_grades, 1),
            'assignment_score': np.round(assignment_scores, 1),
            'quiz_score': np.round(quiz_scores, 1),
            'midterm_score': np.round(midterm_scores, 1),
            'attendance_percentage': np.round(attendance, 1),
            'lms_login_count': np.round(lms_login, 0).astype(int),
            'lms_activity_count': np.round(lms_activity, 0).astype(int),
            'study_hours': np.round(study_hours, 1),
            'performance_status': performance_status
        })
        
        logger.info(f"Generated {len(data)} synthetic records")
        logger.info(f"Class distribution: {data['performance_status'].value_counts().to_dict()}")
        
        # Save the dataset
        output_path = RAW_DATA_DIR / 'student_performance.csv'
        data.to_csv(output_path, index=False)
        logger.info(f"Synthetic dataset saved to {output_path}")
        
        self.data = data
        self.file_path = output_path
        return data
    
    def get_data_info(self):
        """Get dataset information"""
        if self.data is None:
            return None
        
        target_dist = {}
        if 'performance_status' in self.data.columns:
            target_dist = self.data['performance_status'].value_counts().to_dict()
        
        return {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.astype(str).to_dict(),
            'target_distribution': target_dist
        }