# src/features/selection.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from src.utils.logger import logger
from src.config import TOP_N_FEATURES

class FeatureSelector:
    """Feature selection using mutual information"""
    
    def __init__(self, n_features=TOP_N_FEATURES):
        self.n_features = n_features
        self.selector = None
        self.selected_features = None
        self.feature_scores = None
        self.feature_names = None
        self.selected_indices = None
        
    def select_features(self, X, y, feature_names=None):
        """Select top features using mutual information"""
        logger.info(f"Selecting top {self.n_features} features...")
        
        self.feature_names = feature_names
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create feature scores with proper names
        if feature_names is not None and len(feature_names) == len(mi_scores):
            self.feature_scores = pd.DataFrame({
                'feature': feature_names,
                'score': mi_scores
            }).sort_values('score', ascending=False)
            logger.info(f"Using feature names: {feature_names[:5]}")
        else:
            self.feature_scores = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(len(mi_scores))],
                'score': mi_scores
            }).sort_values('score', ascending=False)
        
        # Select top N features
        self.selected_features = self.feature_scores.head(self.n_features)['feature'].tolist()
        
        # Get the indices of selected features
        self.selected_indices = []
        for feat in self.selected_features:
            if feature_names is not None:
                idx = np.where(np.array(feature_names) == feat)[0]
                if len(idx) > 0:
                    self.selected_indices.append(idx[0])
        
        # Create selector
        self.selector = SelectKBest(mutual_info_classif, k=min(self.n_features, X.shape[1]))
        X_selected = self.selector.fit_transform(X, y)
        
        logger.info(f"Selected {len(self.selected_features)} features from {X.shape[1]} total features")
        logger.info(f"Selected features: {self.selected_features}")
        
        return X_selected, self.selected_features
    
    def transform(self, X):
        """Transform data using fitted selector"""
        if self.selector is None:
            raise ValueError("Selector not fitted. Call select_features first.")
        return self.selector.transform(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.feature_scores is None:
            return None
        
        return self.feature_scores.to_dict('records')
    
    def get_selected_features(self):
        """Get list of selected feature names"""
        return self.selected_features
    
    def get_selected_indices(self):
        """Get indices of selected features"""
        return self.selected_indices