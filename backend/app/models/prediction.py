"""Prediction database model."""
from datetime import datetime
from typing import Dict, List, Union
from beanie import Document, Indexed
from bson import ObjectId


class Prediction(Document):
    """Prediction document model."""
    
    model_id: Indexed(ObjectId)
    user_id: Indexed(ObjectId)
    input_data: Union[Dict, List[Dict]]
    output: Union[Dict, List]
    probabilities: Union[Dict, List, None] = None
    timestamp: datetime = datetime.utcnow()
    batch: bool = False
    
    class Settings:
        name = "predictions"
        indexes = [
            "user_id",
            "model_id",
            "timestamp",
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "507f1f77bcf86cd799439011",
                "user_id": "507f1f77bcf86cd799439012",
                "input_data": {"sepal_length": 5.1, "sepal_width": 3.5},
                "output": {"prediction": "setosa"},
                "probabilities": {"setosa": 0.95, "versicolor": 0.03, "virginica": 0.02},
                "timestamp": "2024-01-01T00:00:00",
            }
        }

