"""ML Model database model."""
from datetime import datetime
from typing import Dict, Optional
from beanie import Document, Indexed, PydanticObjectId


class MLModel(Document):
    """ML Model document model."""
    
    job_id: Indexed(PydanticObjectId)
    user_id: Indexed(PydanticObjectId)
    dataset_id: PydanticObjectId
    model_path: str
    model_type: str
    problem_type: str
    target_column: str
    feature_columns: list
    metrics: Dict
    feature_importance: Optional[Dict] = None
    confusion_matrix: Optional[list] = None
    parameters: Dict = {}
    created_at: datetime = datetime.utcnow()
    version: int = 1
    
    class Settings:
        name = "ml_models"
        indexes = [
            "user_id",
            "job_id",
            "model_type",
            "created_at",
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "507f1f77bcf86cd799439011",
                "user_id": "507f1f77bcf86cd799439012",
                "dataset_id": "507f1f77bcf86cd799439013",
                "model_path": "/data/models/model_uuid.pkl",
                "model_type": "random_forest",
                "problem_type": "classification",
                "target_column": "species",
                "feature_columns": ["sepal_length", "sepal_width"],
                "metrics": {"accuracy": 0.95, "f1_score": 0.94},
                "version": 1,
            }
        }

