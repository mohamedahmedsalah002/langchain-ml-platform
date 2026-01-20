"""Training job database model."""
from datetime import datetime
from typing import Dict, Optional
from beanie import Document, Indexed, PydanticObjectId


class TrainingJob(Document):
    """Training job document model."""
    
    dataset_id: Indexed(PydanticObjectId)
    user_id: Indexed(PydanticObjectId)
    model_type: str  # logistic_regression, random_forest, xgboost, etc.
    problem_type: str  # classification, regression
    target_column: str
    feature_columns: list
    status: str = "pending"  # pending, running, completed, failed
    parameters: Dict = {}
    progress: int = 0  # 0-100
    current_stage: Optional[str] = None
    metrics: Optional[Dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    celery_task_id: Optional[str] = None
    
    class Settings:
        name = "training_jobs"
        indexes = [
            "user_id",
            "dataset_id",
            "status",
            "started_at",
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "507f1f77bcf86cd799439011",
                "user_id": "507f1f77bcf86cd799439012",
                "model_type": "random_forest",
                "problem_type": "classification",
                "target_column": "species",
                "feature_columns": ["sepal_length", "sepal_width"],
                "status": "completed",
                "parameters": {"n_estimators": 100, "max_depth": 10},
                "progress": 100,
                "metrics": {"accuracy": 0.95, "f1_score": 0.94},
            }
        }

