"""Database models package using Beanie ODM.

Contains MongoDB document models:
- User: User authentication and profile
- Dataset: Uploaded dataset metadata
- TrainingJob: ML training job tracking
- MLModel: Trained model metadata
- Prediction: Prediction history
- ChatSession: LangChain conversation history
"""

from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.ml_model import MLModel
from app.models.prediction import Prediction
from app.models.chat_session import ChatSession

__all__ = [
    'User',
    'Dataset',
    'TrainingJob',
    'MLModel',
    'Prediction',
    'ChatSession',
]

