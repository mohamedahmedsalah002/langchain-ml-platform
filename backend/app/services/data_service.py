"""Data processing and management service."""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split


class DataService:
    """Service for data loading, processing, and profiling."""
    
    @staticmethod
    def load_dataset(filepath: str) -> pd.DataFrame:
        """Load dataset from file based on extension."""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            return pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    @staticmethod
    def profile_dataset(filepath: str) -> Dict[str, Any]:
        """Generate comprehensive profile of a dataset."""
        df = DataService.load_dataset(filepath)
        
        num_rows, num_columns = df.shape
        
        # Column information
        column_info = {}
        missing_values = {}
        statistics = {}
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing_count = int(df[col].isnull().sum())
            missing_pct = (missing_count / len(df)) * 100
            
            column_info[col] = {
                'dtype': dtype,
                'unique_values': int(df[col].nunique()),
                'missing_count': missing_count,
                'missing_percentage': round(missing_pct, 2)
            }
            
            missing_values[col] = missing_count
            
            # Statistics for numerical columns
            if pd.api.types.is_numeric_dtype(df[col]):
                statistics[col] = {
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'median': float(df[col].median()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'q25': float(df[col].quantile(0.25)) if not df[col].isnull().all() else None,
                    'q75': float(df[col].quantile(0.75)) if not df[col].isnull().all() else None,
                }
            # Statistics for categorical columns
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                value_counts = df[col].value_counts().head(10).to_dict()
                statistics[col] = {
                    'top_values': {str(k): int(v) for k, v in value_counts.items()},
                    'unique_count': int(df[col].nunique())
                }
        
        return {
            'num_rows': num_rows,
            'num_columns': num_columns,
            'column_info': column_info,
            'missing_values': missing_values,
            'statistics': statistics
        }
    
    @staticmethod
    def prepare_data_for_training(
        filepath: str,
        target_column: str,
        feature_columns: list,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for model training."""
        df = DataService.load_dataset(filepath)
        
        # Select features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Handle missing values (simple strategy: drop rows with missing values)
        combined = pd.concat([X, y], axis=1)
        combined = combined.dropna()
        X = combined[feature_columns]
        y = combined[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if DataService.is_classification(y) else None
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def is_classification(y: pd.Series) -> bool:
        """Determine if the problem is classification or regression."""
        # If target is object/categorical or has few unique values, it's classification
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            return True
        
        unique_ratio = y.nunique() / len(y)
        return unique_ratio < 0.05  # If less than 5% unique values, treat as classification
    
    @staticmethod
    def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        X_encoded = X.copy()
        categorical_columns = X_encoded.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_columns) > 0:
            X_encoded = pd.get_dummies(X_encoded, columns=categorical_columns, drop_first=True)
        
        return X_encoded

