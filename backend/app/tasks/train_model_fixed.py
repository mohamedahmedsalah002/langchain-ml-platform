"""
Fixed Celery task for text classification to avoid feature mismatch issues.
"""
import os
import uuid
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from celery import Task
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from bson import ObjectId

from app.celery_app import celery_app
from app.config import settings
from app.models.training_job import TrainingJob
from app.models.ml_model import MLModel
from app.models.user import User
from app.models.dataset import Dataset
from app.models.prediction import Prediction
from app.models.chat_session import ChatSession


class TextClassificationTask(Task):
    """Task for text classification with proper pipeline handling."""
    
    _db_initialized = False
    
    async def init_db(self):
        """Initialize database connection for Celery workers."""
        if not self._db_initialized:
            client = AsyncIOMotorClient(settings.MONGODB_URL)
            database = client[settings.MONGODB_DB_NAME]
            
            await init_beanie(
                database=database,
                document_models=[
                    User, Dataset, TrainingJob, MLModel, Prediction, ChatSession
                ]
            )
            self._db_initialized = True


@celery_app.task(bind=True, base=TextClassificationTask)
def train_text_classification_task(
    self,
    job_id: str,
    dataset_path: str,
    target_column: str,
    text_column: str,
    model_name: str,
    parameters: dict = None
):
    """Train text classification model with proper pipeline handling."""
    import asyncio
    
    async def _train():
        # Initialize database
        await self.init_db()
        
        try:
            # Get training job
            job = await TrainingJob.get(ObjectId(job_id))
            job.status = "running"
            job.started_at = datetime.utcnow()
            job.progress = 0
            job.current_stage = "Loading data"
            await job.save()
            
            # Stage 1: Load data (0-10%)
            self.update_state(state='PROGRESS', meta={'progress': 5, 'stage': 'Loading dataset'})
            
            try:
                df = pd.read_csv(dataset_path)
            except Exception as e:
                raise Exception(f"Failed to load dataset: {str(e)}")
                
            # Clean data
            df = df.dropna(subset=[text_column, target_column])
            
            if len(df) < 10:
                raise Exception(f"Insufficient data: only {len(df)} valid records found")
            
            job.progress = 10
            job.current_stage = "Preparing data"
            await job.save()
            self.update_state(state='PROGRESS', meta={'progress': 10, 'stage': 'Preparing data'})
            
            # Stage 2: Balance dataset for better training (10-20%)
            # Use smart balancing - ignore very small classes to get better training data
            category_counts = df[target_column].value_counts()
            
            # Filter out classes with too few samples (less than 10)
            large_classes = category_counts[category_counts >= 10].index.tolist()
            
            if len(large_classes) < 2:
                # If we have fewer than 2 large classes, use original dataset
                print("Using original dataset - too few large classes for balancing")
                df_balanced = df.sample(min(1000, len(df)), random_state=42)  # Sample up to 1000
            else:
                # Balance using only the larger classes
                balanced_data = []
                sample_size = min(200, category_counts[large_classes].min())  # Up to 200 per class
                
                for category in large_classes:
                    category_data = df[df[target_column] == category]
                    n_samples = min(sample_size, len(category_data))
                    balanced_data.append(category_data.sample(n=n_samples, random_state=42))
                
                df_balanced = pd.concat(balanced_data, ignore_index=True)
            
            print(f"Balanced dataset: {len(df_balanced)} samples, {df_balanced[target_column].nunique()} classes")
            
            job.progress = 20
            job.current_stage = "Setting up ML pipeline"
            await job.save()
            self.update_state(state='PROGRESS', meta={'progress': 20, 'stage': 'Setting up pipeline'})
            
            # Stage 3: Prepare features and target (20-30%)
            X = df_balanced[text_column].astype(str)  # Ensure text
            y = df_balanced[target_column].astype(str)  # Ensure string labels
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            job.progress = 30
            job.current_stage = "Splitting data"
            await job.save()
            self.update_state(state='PROGRESS', meta={'progress': 30, 'stage': 'Splitting data'})
            
            # Stage 4: Split data (30-40%)
            # Adjust test_size based on number of samples and classes
            n_classes = len(np.unique(y_encoded))
            n_samples = len(X)
            min_test_samples = max(n_classes, 10)  # Need at least as many test samples as classes
            
            # Calculate appropriate test_size
            if n_samples < min_test_samples * 5:  # If dataset is too small
                test_size = max(0.1, min_test_samples / n_samples)  # Use smaller test split
                stratify = None  # Don't stratify if too small
            else:
                test_size = 0.2
                stratify = y_encoded if n_classes > 1 else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=test_size, 
                random_state=42, 
                stratify=stratify
            )
            
            job.progress = 40
            job.current_stage = "Creating ML pipeline"
            await job.save()
            self.update_state(state='PROGRESS', meta={'progress': 40, 'stage': 'Creating pipeline'})
            
            # Stage 5: Create ML Pipeline (40-50%)
            # Use Pipeline to avoid feature mismatch issues
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=min(2000, len(X_train)),  # Limit features based on data size
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_df=0.9,
                    min_df=1
                )),
                ('classifier', LogisticRegression(
                    random_state=42,
                    max_iter=300,
                    multi_class='ovr',
                    solver='liblinear',
                    C=parameters.get('C', 1.0) if parameters else 1.0
                ))
            ])
            
            job.progress = 50
            job.current_stage = "Training model"
            await job.save()
            self.update_state(state='PROGRESS', meta={'progress': 60, 'stage': 'Training model'})
            
            # Stage 6: Train model (50-80%)
            pipeline.fit(X_train, y_train)
            
            job.progress = 80
            job.current_stage = "Evaluating model"
            await job.save()
            self.update_state(state='PROGRESS', meta={'progress': 85, 'stage': 'Evaluating model'})
            
            # Stage 7: Evaluate model (80-90%)
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
            
            job.progress = 90
            job.current_stage = "Saving model"
            await job.save()
            self.update_state(state='PROGRESS', meta={'progress': 95, 'stage': 'Saving model'})
            
            # Stage 8: Save model (90-100%)
            model_dir = "/app/data/models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_id = str(uuid.uuid4())
            pipeline_path = f"{model_dir}/{model_id}_pipeline.pkl"
            labels_path = f"{model_dir}/{model_id}_labels.pkl"
            
            # Save pipeline and label encoder
            joblib.dump(pipeline, pipeline_path)
            joblib.dump(label_encoder, labels_path)
            
            # Create ML model record
            ml_model = MLModel(
                job_id=job.id,
                user_id=job.user_id,
                dataset_id=job.dataset_id,
                model_path=pipeline_path,
                model_type="logistic_regression",
                problem_type="classification",
                target_column=target_column,
                feature_columns=[text_column],
                metrics=metrics,
                parameters=parameters or {}
            )
            await ml_model.insert()
            
            # Update job to completed
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.progress = 100
            job.current_stage = "Completed"
            job.metrics = metrics
            await job.save()
            
            self.update_state(
                state='SUCCESS',
                meta={
                    'progress': 100,
                    'stage': 'Completed',
                    'model_id': str(ml_model.id),
                    'metrics': metrics
                }
            )
            
            return {
                'status': 'completed',
                'model_id': str(ml_model.id),
                'metrics': metrics
            }
            
        except Exception as e:
            # Handle errors
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            await job.save()
            
            self.update_state(
                state='FAILURE',
                meta={'error': str(e), 'progress': job.progress}
            )
            
            raise Exception(f"Training failed: {str(e)}")
    
    # Run the async function
    return asyncio.get_event_loop().run_until_complete(_train())