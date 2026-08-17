# src/models/explainability.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from src.utils.logger import logger

class ModelExplainer:
    """Explain model predictions and feature importance"""
    
    def __init__(self):
        self.feature_importances = None
        self.feature_names = None
        self.model = None
        
    def set_model(self, model):
        """Set the model to explain"""
        self.model = model
        return self
    
    def set_feature_names(self, feature_names):
        """Set feature names - ensure uniqueness"""
        if feature_names:
            # Remove duplicates while preserving order
            seen = set()
            unique_names = []
            for name in feature_names:
                if name not in seen:
                    seen.add(name)
                    unique_names.append(name)
            self.feature_names = unique_names
            logger.info(f"Feature names set ({len(self.feature_names)}): {self.feature_names}")
        else:
            self.feature_names = []
            logger.warning("No feature names provided")
        return self
        
    def get_feature_importance(self, model=None, feature_names=None):
        """Get feature importance from the model - ONLY for the features the model was trained on"""
        if model is not None:
            self.model = model
        
        if feature_names is not None:
            self.set_feature_names(feature_names)
        
        if self.model is None:
            logger.warning("No model set for explainability")
            return {}
        
        importance_dict = {}
        
        # Tree-based models
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            logger.info(f"Model has {len(importances)} feature importances")
            
            # Use the feature names that match the model's features
            if self.feature_names is not None and len(self.feature_names) == len(importances):
                # Use actual feature names
                importance_dict = dict(zip(self.feature_names, importances))
                logger.info(f"Using {len(importance_dict)} feature names: {list(self.feature_names)}")
            else:
                # If feature names don't match, try to use what we have
                if self.feature_names is not None and len(self.feature_names) > 0:
                    # Use the first N feature names
                    names_to_use = self.feature_names[:len(importances)]
                    if len(names_to_use) == len(importances):
                        importance_dict = dict(zip(names_to_use, importances))
                    else:
                        # Pad with generic names
                        names = names_to_use + [f'Feature_{i+1}' for i in range(len(importances) - len(names_to_use))]
                        importance_dict = dict(zip(names, importances))
                else:
                    importance_dict = {f'Feature_{i+1}': imp for i, imp in enumerate(importances)}
        
        # Linear models
        elif hasattr(self.model, 'coef_'):
            if len(self.model.coef_.shape) > 1:
                coefficients = np.abs(self.model.coef_[0])
            else:
                coefficients = np.abs(self.model.coef_)
            
            if self.feature_names is not None and len(self.feature_names) == len(coefficients):
                importance_dict = dict(zip(self.feature_names, coefficients))
            else:
                if self.feature_names is not None and len(self.feature_names) > 0:
                    names_to_use = self.feature_names[:len(coefficients)]
                    if len(names_to_use) == len(coefficients):
                        importance_dict = dict(zip(names_to_use, coefficients))
                    else:
                        names = names_to_use + [f'Feature_{i+1}' for i in range(len(coefficients) - len(names_to_use))]
                        importance_dict = dict(zip(names, coefficients))
                else:
                    importance_dict = {f'Feature_{i+1}': imp for i, imp in enumerate(coefficients)}
        
        # Sort by importance
        self.feature_importances = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Log top features
        top_items = list(self.feature_importances.items())[:5]
        logger.info(f"Top 5 features: {top_items}")
        
        return self.feature_importances
    
    def get_top_features(self, n=10):
        """Get top N most important features with actual names - ensure uniqueness"""
        if self.feature_importances is None:
            return []
        
        items = list(self.feature_importances.items())[:n]
        # Ensure unique features
        seen = set()
        unique_items = []
        for name, value in items:
            if name not in seen:
                seen.add(name)
                unique_items.append((name, value))
        
        return [{'feature': name, 'importance': float(value)} for name, value in unique_items]
    
    def explain_prediction(self, features, prediction_result):
        """Explain a single prediction"""
        explanation = {
            'student': features.get('student_id', 'Unknown'),
            'prediction': prediction_result.get('prediction', 'Unknown'),
            'confidence': prediction_result.get('confidence', 0),
            'risk_level': prediction_result.get('risk_level', 'Unknown'),
            'factors': [],
            'recommendation': prediction_result.get('recommendation', '')
        }
        
        # Add feature values to explanation
        if self.feature_importances is not None:
            seen = set()
            for feature, importance in list(self.feature_importances.items())[:5]:
                if feature not in seen and feature in features:
                    seen.add(feature)
                    explanation['factors'].append({
                        'feature': feature,
                        'value': features[feature],
                        'importance': float(importance),
                        'impact': self._get_impact_level(importance)
                    })
        
        return explanation
    
    def _get_impact_level(self, importance):
        """Get impact level based on importance score"""
        if importance > 0.3:
            return 'HIGH'
        elif importance > 0.15:
            return 'MEDIUM'
        else:
            return 'LOW'