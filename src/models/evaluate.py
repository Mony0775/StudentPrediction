# src/models/evaluate.py
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from src.utils.logger import logger
from src.config import MODELS_DIR, METRICS_PATH

class ModelEvaluator:
    """Evaluate trained models"""
    
    def __init__(self):
        self.metrics = {}
        self.confusion_matrices = {}
        self.classification_reports = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        logger.info(f"Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }
        
        # Add ROC-AUC if probability available
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba[:, 1]))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba, multi_class='ovr'))
        except:
            metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.metrics[model_name] = metrics
        self.confusion_matrices[model_name] = cm.tolist()
        self.classification_reports[model_name] = report
        
        logger.info(f"{model_name} metrics: {metrics}")
        return metrics
    
    def evaluate_all(self, models, X_test, y_test):
        """Evaluate all models"""
        results = {}
        
        for name, model in models.items():
            results[name] = self.evaluate_model(model, X_test, y_test, name)
        
        return results
    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_data = {
            'metrics': self.metrics,
            'confusion_matrices': self.confusion_matrices,
            'classification_reports': self.classification_reports
        }
        
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics saved to {METRICS_PATH}")
        return metrics_data
    
    def get_comparison_table(self):
        """Get comparison table as DataFrame with models as rows"""
        if not self.metrics:
            return pd.DataFrame()
        
        # Create DataFrame with models as rows and metrics as columns
        data = []
        for model_name, metrics in self.metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.set_index('Model')
        return df.round(4)  