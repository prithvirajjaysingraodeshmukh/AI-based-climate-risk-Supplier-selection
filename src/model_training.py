"""
ML Model Training Module
Trains and evaluates Random Forest classifier for climate risk prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

class ClimateRiskPredictor:
    """Trains and evaluates climate risk prediction model"""
    
    def __init__(self, n_estimators=100, random_state=42, max_depth=10):
        """
        Initialize Random Forest Classifier
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            class_weight='balanced'  # Handle class imbalance
        )
        self.feature_importance_ = None
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training Random Forest Classifier...")
        self.model.fit(X_train, y_train)
        self.feature_importance_ = pd.DataFrame({
            'feature': X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        self.is_trained = True
        print("Model training completed!")
        return self
    
    def predict(self, X):
        """Predict climate risk levels"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probability of each class"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, label_mapping):
        """
        Evaluate model performance
        Returns evaluation metrics and confusion matrix
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Predictions
        y_pred = self.predict(X_test)
        
        # Convert encoded labels back to original labels
        if label_mapping:
            y_test_labels = [label_mapping[label] for label in y_test]
            y_pred_labels = [label_mapping[label] for label in y_pred]
        else:
            y_test_labels = y_test
            y_pred_labels = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=list(label_mapping.values()) if label_mapping else None)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(report)
        print("="*50 + "\n")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_test': y_test_labels,
            'y_pred': y_pred_labels
        }
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.feature_importance_ is None:
            raise ValueError("Model must be trained first")
        return self.feature_importance_
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

