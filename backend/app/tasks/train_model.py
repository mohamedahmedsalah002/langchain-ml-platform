"""Celery task for model training."""
import os
import uuid
import joblib
from datetime import datetime
from celery import Task
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
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
from app.services.data_service import DataService


class TrainingTask(Task):
    """Base task with database connection."""
    
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


@celery_app.task(bind=True, base=TrainingTask)
def train_model_task(
    self,
    job_id: str,
    dataset_path: str,
    target_column: str,
    feature_columns: list,
    model_type: str,
    problem_type: str,
    parameters: dict,
    test_size: float = 0.2
):
    """Train a machine learning model asynchronously."""
    import asyncio
    
    async def _train():
        # Initialize database
        await self.init_db()
        
        try:
            # Update job status to running
            job = await TrainingJob.get(ObjectId(job_id))
            job.status = "running"
            job.started_at = datetime.utcnow()
            job.progress = 0
            job.current_stage = "Loading data"
            await job.save()
            
            # Stage 1: Load and prepare data (0-20%)
            self.update_state(state='PROGRESS', meta={'progress': 10, 'stage': 'Loading data'})
            X_train, X_test, y_train, y_test = DataService.prepare_data_for_training(
                dataset_path, target_column, feature_columns, test_size
            )
            
            # Encode categorical features
            X_train = DataService.encode_categorical_features(X_train)
            X_test = DataService.encode_categorical_features(X_test)
            
            job.progress = 20
            job.current_stage = "Initializing model"
            await job.save()
            
            # Stage 2: Initialize model (20-30%)
            self.update_state(state='PROGRESS', meta={'progress': 25, 'stage': 'Initializing model'})
            model = get_model(model_type, problem_type, parameters)
            
            job.progress = 30
            job.current_stage = "Training model"
            await job.save()
            
            # Stage 3: Train model (30-80%)
            self.update_state(state='PROGRESS', meta={'progress': 50, 'stage': 'Training model'})
            model.fit(X_train, y_train)
            
            job.progress = 80
            job.current_stage = "Evaluating model"
            await job.save()
            
            # Stage 4: Evaluate model (80-90%)
            self.update_state(state='PROGRESS', meta={'progress': 85, 'stage': 'Evaluating model'})
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if problem_type == "classification":
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                }
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                confusion_mat = cm.tolist()
            else:
                metrics = {
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    'mae': float(mean_absolute_error(y_test, y_pred)),
                    'r2': float(r2_score(y_test, y_pred))
                }
                confusion_mat = None
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = {
                    feature: float(importance)
                    for feature, importance in zip(X_train.columns, model.feature_importances_)
                }
            elif hasattr(model, 'coef_'):
                feature_importance = {
                    feature: float(coef)
                    for feature, coef in zip(X_train.columns, model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
                }
            
            job.progress = 90
            job.current_stage = "Saving model"
            await job.save()
            
            # Stage 5: Save model (90-100%)
            self.update_state(state='PROGRESS', meta={'progress': 95, 'stage': 'Saving model'})
            
            # Save model to disk
            os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)
            model_id = str(uuid.uuid4())
            model_path = os.path.join(settings.MODEL_STORAGE_PATH, f"{model_id}.pkl")
            joblib.dump(model, model_path)
            
            # Create MLModel record
            ml_model = MLModel(
                job_id=ObjectId(job_id),
                user_id=job.user_id,
                dataset_id=job.dataset_id,
                model_path=model_path,
                model_type=model_type,
                problem_type=problem_type,
                target_column=target_column,
                feature_columns=list(X_train.columns),  # Use encoded column names
                metrics=metrics,
                feature_importance=feature_importance,
                confusion_matrix=confusion_mat,
                parameters=parameters,
                created_at=datetime.utcnow()
            )
            await ml_model.insert()
            
            # Update job as completed
            job.status = "completed"
            job.progress = 100
            job.current_stage = "Completed"
            job.metrics = metrics
            job.completed_at = datetime.utcnow()
            await job.save()
            
            self.update_state(state='SUCCESS', meta={'progress': 100, 'stage': 'Completed'})
            
            return {
                'job_id': job_id,
                'model_id': str(ml_model.id),
                'metrics': metrics,
                'status': 'completed'
            }
        
        except Exception as e:
            # Update job as failed
            job = await TrainingJob.get(ObjectId(job_id))
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await job.save()
            
            raise
    
    # Run async function
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_train())


def get_model(model_type: str, problem_type: str, parameters: dict):
    """Get the appropriate model based on type and problem."""
    if problem_type == "classification":
        if model_type == "logistic_regression":
            return LogisticRegression(**parameters)
        elif model_type == "random_forest":
            return RandomForestClassifier(**parameters)
        elif model_type == "xgboost":
            return XGBClassifier(**parameters)
        elif model_type == "svm":
            return SVC(**parameters, probability=True)
        elif model_type == "neural_network":
            return MLPClassifier(**parameters)
        else:
            raise ValueError(f"Unknown classification model type: {model_type}")
    else:  # regression
        if model_type == "random_forest":
            return RandomForestRegressor(**parameters)
        elif model_type == "xgboost":
            return XGBRegressor(**parameters)
        elif model_type == "svm":
            return SVR(**parameters)
        elif model_type == "neural_network":
            return MLPRegressor(**parameters)
        else:
            raise ValueError(f"Unknown regression model type: {model_type}")

