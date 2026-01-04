"""
Data Preprocessing Module
Handles missing values, normalization, and encoding for the supplier dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Handles all data preprocessing steps"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.scaler_fitted = False
        
    def load_data(self, file_path):
        """Load dataset from CSV file"""
        df = pd.read_csv(file_path)
        return df
    
    def handle_missing_values(self, df, strategy='median'):
        """
        Handle missing values in the dataset
        strategy: 'median', 'mean', or 'mode'
        """
        df_processed = df.copy()
        
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            missing_count = df_processed[col].isna().sum()
            if missing_count > 0:
                if strategy == 'median':
                    fill_value = df_processed[col].median()
                elif strategy == 'mean':
                    fill_value = df_processed[col].mean()
                elif strategy == 'mode':
                    fill_value = df_processed[col].mode()[0]
                else:
                    fill_value = 0
                
                df_processed[col].fillna(fill_value, inplace=True)
                print(f"Filled {missing_count} missing values in {col} with {strategy}: {fill_value:.2f}")
        
        return df_processed
    
    def encode_categorical_features(self, df):
        """Encode categorical features (country, supplier_name)"""
        df_encoded = df.copy()
        
        # One-hot encode country (for feature engineering, but we'll use numeric features for ML)
        # We'll keep country as is for now, but could encode if needed
        
        return df_encoded
    
    def prepare_features(self, df):
        """
        Prepare feature set for ML model
        Returns feature matrix and target vector
        """
        # Select climate features for prediction
        climate_features = [
            'avg_temperature_celsius',
            'rainfall_mm_per_year',
            'extreme_events_per_year'
        ]
        
        # Prepare feature matrix
        X = df[climate_features].copy()
        self.feature_columns = climate_features
        
        # Prepare target (climate risk level)
        y = df['climate_risk_level'].copy()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded, y
    
    def normalize_features(self, X, fit=True):
        """Normalize features using StandardScaler"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.scaler_fitted = True
        else:
            if not self.scaler_fitted:
                raise ValueError("Scaler must be fitted first. Call with fit=True")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def get_label_mapping(self):
        """Get mapping of encoded labels to original labels"""
        if hasattr(self.label_encoder, 'classes_'):
            return dict(enumerate(self.label_encoder.classes_))
        return {}
    
    def preprocess_pipeline(self, file_path, normalize=True, test_size=0.2):
        """
        Complete preprocessing pipeline
        Returns: X_train, X_test, y_train, y_test, df_processed, label_mapping
        """
        # Load data
        df = self.load_data(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Handle missing values
        df_processed = self.handle_missing_values(df, strategy='median')
        
        # Encode categorical features (if needed)
        df_processed = self.encode_categorical_features(df_processed)
        
        # Prepare features and target
        X, y_encoded, y_original = self.prepare_features(df_processed)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution:\n{y_original.value_counts()}")
        
        # Normalize features
        if normalize:
            X_scaled = self.normalize_features(X, fit=True)
            X = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(
            X, y_encoded, test_size=test_size
        )
        
        label_mapping = self.get_label_mapping()
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test, df_processed, label_mapping

