# src/utils/helpers.py
import json
import pandas as pd
import numpy as np
from src.utils.logger import logger

def save_json(data, filepath):
    """Save data as JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        return False

def load_json(filepath):
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return None

def get_performance_status(probability):
    """Map probability to risk level"""
    if probability < 0.30:
        return 'SAFE', 'LOW'
    elif probability < 0.60:
        return 'AT_RISK', 'MEDIUM'
    elif probability < 0.80:
        return 'AT_RISK', 'HIGH'
    else:
        return 'FAIL', 'CRITICAL'

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }