"""Database connection and initialization."""
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.config import settings
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.ml_model import MLModel
from app.models.prediction import Prediction
from app.models.chat_session import ChatSession


async def init_db():
    """Initialize database connection and Beanie ODM."""
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    database = client[settings.MONGODB_DB_NAME]
    
    await init_beanie(
        database=database,
        document_models=[
            User,
            Dataset,
            TrainingJob,
            MLModel,
            Prediction,
            ChatSession,
        ]
    )
    
    print(f"Connected to MongoDB: {settings.MONGODB_DB_NAME}")

