"""Dataset database model."""
from datetime import datetime
from typing import Dict, Optional
from beanie import Document, Indexed, Link
from pydantic import BaseModel
from bson import ObjectId


class Dataset(Document):
    """Dataset document model."""
    
    user_id: Indexed(ObjectId)
    filename: str
    filepath: str
    size: int  # File size in bytes
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    column_types: Optional[Dict[str, str]] = None
    column_names: Optional[list] = None
    upload_date: datetime = datetime.utcnow()
    status: str = "uploaded"  # uploaded, processing, ready, error
    error_message: Optional[str] = None
    
    class Settings:
        name = "datasets"
        indexes = [
            "user_id",
            "status",
            "upload_date",
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "filename": "iris.csv",
                "filepath": "/data/datasets/uuid.csv",
                "size": 5120,
                "num_rows": 150,
                "num_columns": 5,
                "column_types": {"sepal_length": "float64", "species": "object"},
                "status": "ready",
            }
        }

