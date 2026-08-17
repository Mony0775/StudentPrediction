# src/models/predict.py
import pickle
import pandas as pd
import numpy as np
from src.utils.logger import logger
from src.config import (
    MODELS_DIR, BEST_MODEL_PATH, PREPROCESSOR_PATH,
    FEATURE_SELECTOR_PATH, LABEL_ENCODER_PATH,
    LOGISTIC_REGRESSION_PATH, DECISION_TREE_PATH,
    RANDOM_FOREST_PATH, NAIVE_BAYES_PATH,
    RISK_THRESHOLDS, RECOMMENDATIONS
)

class Predictor:
    """Handle predictions with the best model"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_names = None
        self.is_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load trained models from .pkl files"""
        try:
            # Load best model
            if BEST_MODEL_PATH.exists():
                with open(BEST_MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Loaded best model from .pkl")
                self.is_loaded = True
            else:
                # Try to load any available model
                model_paths = {
                    'Logistic Regression': LOGISTIC_REGRESSION_PATH,
                    'Decision Tree': DECISION_TREE_PATH,
                    'Random Forest': RANDOM_FOREST_PATH,
                    'Naive Bayes': NAIVE_BAYES_PATH
                }
                
                for name, path in model_paths.items():
                    if path.exists():
                        with open(path, 'rb') as f:
                            self.model = pickle.load(f)
                        logger.info(f"Loaded {name} from .pkl")
                        self.is_loaded = True
                        break
                
                if not self.is_loaded:
                    logger.warning("No model found. Please train models first.")
                    self.is_loaded = False
            
            # Load preprocessor
            if PREPROCESSOR_PATH.exists():
                with open(PREPROCESSOR_PATH, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                logger.info("Loaded preprocessor from .pkl")
            
            # Load label encoder
            if LABEL_ENCODER_PATH.exists():
                with open(LABEL_ENCODER_PATH, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Loaded label encoder from .pkl")
            
            # Load feature selector
            if FEATURE_SELECTOR_PATH.exists():
                with open(FEATURE_SELECTOR_PATH, 'rb') as f:
                    self.feature_selector = pickle.load(f)
                logger.info("Loaded feature selector from .pkl")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
    
    def check_model_loaded(self):
        """Check if model is loaded and raise error if not"""
        if not self.is_loaded or self.model is None:
            self.load_models()
            if not self.is_loaded or self.model is None:
                raise ValueError("Model not trained. Please train models first using the 'Train Models' button.")
        return True
    
    def preprocess_features(self, df):
        """Preprocess features for prediction"""
        # Remove student_id if present
        if 'student_id' in df.columns:
            df = df.drop(columns=['student_id'])
        
        # Handle preprocessing
        if self.preprocessor is not None:
            try:
                X_processed = self.preprocessor.transform(df)
            except Exception as e:
                logger.error(f"Preprocessing error: {e}")
                try:
                    X_processed = self.preprocessor.fit_transform(df)
                except:
                    raise ValueError("Could not preprocess data. Please ensure the data format is correct.")
        else:
            X_processed = df.values if hasattr(df, 'values') else df
        
        # Apply feature selection if available
        if self.feature_selector is not None:
            try:
                X_selected = self.feature_selector.transform(X_processed)
                logger.info(f"Applied feature selection, reduced to {X_selected.shape[1]} features")
                return X_selected
            except Exception as e:
                logger.error(f"Feature selection error: {e}")
                return X_processed
        else:
            # Check if we need to select features manually
            expected_features = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None
            if expected_features is not None and X_processed.shape[1] != expected_features:
                logger.warning(f"Feature mismatch: got {X_processed.shape[1]}, expected {expected_features}")
        
        return X_processed
    
    def predict_single(self, features):
        """Predict for a single student"""
        self.check_model_loaded()
        
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features
        
        # Preprocess
        X_processed = self.preprocess_features(df)
        
        # Check feature count
        expected_features = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None
        if expected_features is not None and X_processed.shape[1] != expected_features:
            logger.error(f"Feature mismatch: got {X_processed.shape[1]}, expected {expected_features}")
            raise ValueError(f"Feature mismatch: got {X_processed.shape[1]} features, expected {expected_features}. Please ensure feature selection is applied.")
        
        # Predict
        try:
            prediction = self.model.predict(X_processed)
            probability = self.model.predict_proba(X_processed)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")
        
        # Decode prediction
        if self.label_encoder is not None:
            pred_label = self.label_encoder.inverse_transform(prediction)[0]
        else:
            pred_label = prediction[0]
        
        # Get probabilities for each class
        prob_class_0 = probability[0][0]
        prob_class_1 = probability[0][1]
        
        # Determine which class is "FAIL" and which is "PASS"
        if self.label_encoder is not None:
            classes = self.label_encoder.classes_
            fail_idx = np.where(classes == 'FAIL')[0][0] if 'FAIL' in classes else 1
            pass_idx = np.where(classes == 'PASS')[0][0] if 'PASS' in classes else 0
        else:
            fail_idx = 1
            pass_idx = 0
        
        # Get failure probability (probability of FAIL class)
        failure_probability = probability[0][fail_idx]
        
        # Get PASS probability
        pass_probability = probability[0][pass_idx]
        
        # Calculate confidence as the max probability
        confidence = float(max(probability[0]))
        
        # Get risk level based on failure probability
        risk_level = self.get_risk_level(failure_probability)
        
        # Get recommendation
        recommendation = self.get_recommendation(risk_level)
        
        # Get important factors
        important_factors = self.get_important_factors(features)
        
        return {
            'prediction': pred_label,
            'probability': probability.tolist(),
            'confidence': confidence,
            'pass_probability': float(pass_probability),
            'failure_probability': float(failure_probability),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'important_factors': important_factors
        }
    
    def predict_batch(self, data):
        """Predict for multiple students"""
        self.check_model_loaded()
        
        results = []
        df = data.copy()
        
        # Store student IDs if present
        student_ids = df['student_id'].tolist() if 'student_id' in df.columns else None
        
        # Preprocess
        X_processed = self.preprocess_features(df)
        
        # Check feature count
        expected_features = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None
        if expected_features is not None and X_processed.shape[1] != expected_features:
            logger.error(f"Feature mismatch: got {X_processed.shape[1]}, expected {expected_features}")
            raise ValueError(f"Feature mismatch: got {X_processed.shape[1]} features, expected {expected_features}")
        
        # Predict
        try:
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise ValueError(f"Batch prediction failed: {str(e)}")
        
        # Determine class indices
        if self.label_encoder is not None:
            classes = self.label_encoder.classes_
            fail_idx = np.where(classes == 'FAIL')[0][0] if 'FAIL' in classes else 1
            pass_idx = np.where(classes == 'PASS')[0][0] if 'PASS' in classes else 0
        else:
            fail_idx = 1
            pass_idx = 0
        
        # Process each prediction
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if self.label_encoder is not None:
                pred_label = self.label_encoder.inverse_transform([pred])[0]
            else:
                pred_label = pred
            
            failure_probability = prob[fail_idx]
            pass_probability = prob[pass_idx]
            confidence = float(max(prob))
            risk_level = self.get_risk_level(failure_probability)
            
            result = {
                'student_id': student_ids[i] if student_ids else f"S{i+1:04d}",
                'prediction': pred_label,
                'confidence': confidence,
                'pass_probability': float(pass_probability),
                'failure_probability': float(failure_probability),
                'risk_level': risk_level,
                'recommendation': self.get_recommendation(risk_level)
            }
            results.append(result)
        
        return results
    
    def get_risk_level(self, failure_probability):
        """Map failure probability to risk level"""
        if failure_probability <= RISK_THRESHOLDS['LOW']:
            return 'LOW'
        elif failure_probability <= RISK_THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        elif failure_probability <= RISK_THRESHOLDS['HIGH']:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def get_recommendation(self, risk_level):
        """Get recommendation based on risk level"""
        return RECOMMENDATIONS.get(risk_level, "Monitor student performance.")
    
    def get_important_factors(self, features):
        """Get important factors for a prediction"""
        important_factors = []
        
        key_features = {
            'attendance_percentage': {'threshold': 70, 'impact': 'HIGH'},
            'historical_grade': {'threshold': 50, 'impact': 'HIGH'},
            'previous_grade': {'threshold': 50, 'impact': 'MEDIUM'},
            'assignment_score': {'threshold': 50, 'impact': 'HIGH'},
            'quiz_score': {'threshold': 50, 'impact': 'MEDIUM'},
            'midterm_score': {'threshold': 50, 'impact': 'HIGH'},
            'study_hours': {'threshold': 5, 'impact': 'MEDIUM'},
            'lms_login_count': {'threshold': 10, 'impact': 'LOW'},
            'lms_activity_count': {'threshold': 20, 'impact': 'LOW'}
        }
        
        for feature, config in key_features.items():
            if feature in features:
                value = features[feature]
                is_risk = False
                if feature == 'study_hours':
                    is_risk = value < config['threshold']
                elif feature in ['lms_login_count', 'lms_activity_count']:
                    is_risk = value < config['threshold']
                else:
                    is_risk = value < config['threshold']
                
                if is_risk:
                    important_factors.append({
                        'feature': feature.replace('_', ' ').title(),
                        'value': value,
                        'impact': config['impact'],
                        'threshold': config['threshold']
                    })
        
        impact_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        important_factors.sort(key=lambda x: impact_order.get(x['impact'], 3))
        
        return important_factors[:5]