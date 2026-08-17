# src/models/train.py
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import numpy as np
from src.utils.logger import logger
from src.config import (
    TEST_SIZE, RANDOM_STATE, MODELS_DIR, BEST_MODEL_PATH,
    LOGISTIC_REGRESSION_PATH, DECISION_TREE_PATH,
    RANDOM_FOREST_PATH, NAIVE_BAYES_PATH
)

class ModelTrainer:
    """Train and manage multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.best_model_name = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def split_data(self, X, y):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"Data split: {len(self.X_train)} train, {len(self.X_test)} test")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_models(self):
        """Define models to train with proper names"""
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=RANDOM_STATE
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=RANDOM_STATE
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=RANDOM_STATE
            ),
            'Naive Bayes': GaussianNB()
        }
    
    def train_all(self, X, y):
        """Train all models"""
        logger.info("Starting model training...")
        
        # Split data
        self.split_data(X, y)
        
        # Get models
        models = self.get_models()
        self.models = models
        
        # Train each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
            
            # Calculate training score
            train_score = model.score(self.X_train, self.y_train)
            test_score = model.score(self.X_test, self.y_test)
            logger.info(f"{name} - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Save all trained models individually
        self.save_all_models()
        
        # Select best model based on F1 score
        self.select_best_model()
        
        # Save best model
        self.save_best_model()
        
        return self.trained_models
    
    def save_all_models(self):
        """Save all trained models as individual .pkl files"""
        model_paths = {
            'Logistic Regression': LOGISTIC_REGRESSION_PATH,
            'Decision Tree': DECISION_TREE_PATH,
            'Random Forest': RANDOM_FOREST_PATH,
            'Naive Bayes': NAIVE_BAYES_PATH
        }
        
        for name, model in self.trained_models.items():
            if name in model_paths:
                path = model_paths[name]
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} to {path}")
    
    def select_best_model(self, metric='f1_score'):
        """Select the best model based on metric"""
        logger.info(f"Selecting best model based on {metric}...")
        
        best_score = -1
        best_name = None
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(self.X_test)
            score = f1_score(self.y_test, y_pred, average='weighted')
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.trained_models[best_name]
        logger.info(f"Best model: {best_name} with {metric}: {best_score:.3f}")
        
        return best_name, best_score
    
    def save_best_model(self):
        """Save the best model as .pkl file"""
        if self.best_model is not None:
            model_path = BEST_MODEL_PATH
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            logger.info(f"Best model saved to {model_path}")
            return model_path
        return None
    
    def get_best_model(self):
        """Get the best model"""
        return self.best_model, self.best_model_name