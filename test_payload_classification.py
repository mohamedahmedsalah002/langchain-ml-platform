#!/usr/bin/env python3
"""
Comprehensive test of the ML Platform with Payload Classification Dataset
Tests data processing, feature engineering, model training, and evaluation.
"""

import asyncio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PayloadClassificationTest:
    """Test the ML Platform with cybersecurity payload classification."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.label_encoder = None
        self.model = None
        
    async def load_and_analyze_data(self):
        """Load and perform exploratory data analysis."""
        print("🔍 Loading and analyzing payload dataset...")
        
        # Load data
        self.data = pd.read_csv(self.dataset_path)
        print(f"✅ Dataset loaded: {len(self.data)} records")
        
        # Basic info
        print(f"\n📊 Dataset Info:")
        print(f"   - Shape: {self.data.shape}")
        print(f"   - Columns: {list(self.data.columns)}")
        print(f"   - Memory usage: {self.data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Check for missing values
        print(f"\n🔍 Data Quality:")
        missing_values = self.data.isnull().sum()
        print(f"   - Missing values: {missing_values.sum()}")
        for col, missing in missing_values.items():
            if missing > 0:
                print(f"     - {col}: {missing} ({missing/len(self.data)*100:.1f}%)")
        
        # Category distribution
        print(f"\n📈 Category Distribution:")
        category_counts = self.data['category'].value_counts()
        print(category_counts)
        
        # Sample payloads per category
        print(f"\n🎯 Sample Payloads by Category:")
        for category in category_counts.head(5).index:
            sample = self.data[self.data['category'] == category]['payload'].iloc[0]
            print(f"   - {category}: {sample[:100]}...")
        
        return {
            'total_records': len(self.data),
            'categories': len(category_counts),
            'missing_values': missing_values.sum(),
            'category_distribution': category_counts.to_dict()
        }
    
    async def preprocess_data(self):
        """Preprocess the payload data for ML."""
        print("\n🛠️ Preprocessing data for machine learning...")
        
        # Clean data
        self.data = self.data.dropna()  # Remove any missing values
        print(f"   - Records after cleaning: {len(self.data)}")
        
        # Encode categories
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.data['category'])
        print(f"   - Categories encoded: {list(self.label_encoder.classes_)}")
        
        # Feature engineering: Text vectorization
        print("   - Vectorizing payload text with TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit features for faster training
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_df=0.95,  # Remove very common terms
            min_df=2      # Remove very rare terms
        )
        
        X = self.vectorizer.fit_transform(self.data['payload'])
        print(f"   - Feature matrix shape: {X.shape}")
        print(f"   - Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   - Training set: {self.X_train.shape[0]} samples")
        print(f"   - Test set: {self.X_test.shape[0]} samples")
        
        return {
            'train_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0],
            'features': self.X_train.shape[1],
            'classes': len(self.label_encoder.classes_)
        }
    
    async def train_logistic_regression(self):
        """Train logistic regression classifier."""
        print("\n🏋️ Training Logistic Regression model...")
        
        # Configure model
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr',  # One-vs-Rest for multi-class
            solver='liblinear',  # Good for small-medium datasets
            C=1.0  # Regularization strength
        )
        
        # Train model
        print("   - Fitting model to training data...")
        self.model.fit(self.X_train, self.y_train)
        
        # Training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)
        print(f"   - Training accuracy: {train_accuracy:.3f}")
        
        return {
            'model_type': 'logistic_regression',
            'training_accuracy': train_accuracy,
            'n_features': self.X_train.shape[1],
            'n_classes': len(np.unique(self.y_train))
        }
    
    async def evaluate_model(self):
        """Evaluate model performance."""
        print("\n📊 Evaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"   - Test accuracy: {accuracy:.3f}")
        
        # Classification report
        print("\n📈 Classification Report:")
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        print(report)
        
        # Confusion matrix
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Feature importance (top coefficients)
        print("\n🎯 Top Important Features per Class:")
        feature_names = self.vectorizer.get_feature_names_out()
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            if hasattr(self.model, 'coef_'):
                coeffs = self.model.coef_[i] if self.model.coef_.ndim > 1 else self.model.coef_
                top_indices = coeffs.argsort()[-10:][::-1]  # Top 10 features
                print(f"\n   {class_name}:")
                for idx in top_indices[:5]:  # Show top 5
                    print(f"     - {feature_names[idx]}: {coeffs[idx]:.3f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'n_predictions': len(y_pred)
        }
    
    async def test_predictions(self):
        """Test model with sample predictions."""
        print("\n🎯 Testing model predictions...")
        
        # Sample test payloads
        test_payloads = [
            "<script>alert('XSS')</script>",
            "' OR 1=1 --",
            "../../../etc/passwd",
            "{{7*7}}",
            "normal text payload"
        ]
        
        print("   Sample Predictions:")
        for payload in test_payloads:
            # Vectorize
            payload_vector = self.vectorizer.transform([payload])
            
            # Predict
            prediction = self.model.predict(payload_vector)[0]
            probability = self.model.predict_proba(payload_vector)[0]
            confidence = max(probability)
            
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            
            print(f"     - '{payload[:50]}...'")
            print(f"       → Predicted: {predicted_class} (confidence: {confidence:.3f})")
        
        return {
            'test_payloads': test_payloads,
            'predictions_completed': len(test_payloads)
        }
    
    async def run_complete_test(self):
        """Run the complete ML pipeline test."""
        print("🚀 Starting Complete ML Platform Test")
        print("=" * 60)
        
        results = {}
        
        try:
            # Step 1: Load and analyze
            results['data_analysis'] = await self.load_and_analyze_data()
            
            # Step 2: Preprocess
            results['preprocessing'] = await self.preprocess_data()
            
            # Step 3: Train model
            results['training'] = await self.train_logistic_regression()
            
            # Step 4: Evaluate
            results['evaluation'] = await self.evaluate_model()
            
            # Step 5: Test predictions
            results['predictions'] = await self.test_predictions()
            
            print("\n🎉 ML Platform Test Complete!")
            print("=" * 60)
            
            # Summary
            print(f"\n📋 Test Summary:")
            print(f"   ✅ Dataset: {results['data_analysis']['total_records']:,} records")
            print(f"   ✅ Categories: {results['data_analysis']['categories']} classes")
            print(f"   ✅ Features: {results['preprocessing']['features']:,} TF-IDF features")
            print(f"   ✅ Training Accuracy: {results['training']['training_accuracy']:.3f}")
            print(f"   ✅ Test Accuracy: {results['evaluation']['accuracy']:.3f}")
            print(f"   ✅ Predictions: {results['predictions']['predictions_completed']} samples tested")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error during test: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


async def main():
    """Main test function."""
    dataset_path = "/Users/mo/PycharmProjects/JupyterProject1/langchain-ml-platform/data/datasets/payloadbox_combined_payloads.csv"
    
    # Initialize tester
    tester = PayloadClassificationTest(dataset_path)
    
    # Run complete test
    results = await tester.run_complete_test()
    
    print("\n🔍 Want to explore more? Try:")
    print("   - Access the frontend: http://localhost:8501")
    print("   - API documentation: http://localhost:8000/docs")
    print("   - Upload the dataset through the web interface")
    
    return results


if __name__ == "__main__":
    # Run the test
    results = asyncio.run(main())