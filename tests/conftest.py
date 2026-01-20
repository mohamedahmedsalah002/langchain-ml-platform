"""
Test configuration and fixtures for the ML Platform backend tests.
"""
import asyncio
import os
import pytest
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from fastapi.testclient import TestClient
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

# Import the FastAPI app
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from app.main import app
from app.config import get_settings
from app.models.user import User
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.training_job import TrainingJob
from app.models.prediction import Prediction
from app.models.chat_session import ChatSession


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings():
    """Get test settings."""
    os.environ["MONGODB_DB_NAME"] = "test_ml_platform"
    os.environ["TESTING"] = "true"
    return get_settings()


@pytest.fixture
async def test_db(settings):
    """Initialize test database."""
    client = AsyncIOMotorClient(settings.mongodb_url)
    database = client[settings.mongodb_db_name]
    
    # Initialize Beanie with models
    await init_beanie(
        database=database,
        document_models=[
            User,
            Dataset,
            MLModel,
            TrainingJob, 
            Prediction,
            ChatSession
        ]
    )
    
    yield database
    
    # Cleanup: Drop test database
    await client.drop_database(settings.mongodb_db_name)
    client.close()


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def authenticated_user(test_db) -> User:
    """Create authenticated test user."""
    user_data = {
        "email": "test@example.com",
        "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiR0uYvdm5C6",  # "testpass"
        "is_active": True
    }
    user = User(**user_data)
    await user.insert()
    return user


@pytest.fixture
def auth_headers(authenticated_user):
    """Create authentication headers."""
    from app.utils.auth import create_access_token
    access_token = create_access_token(data={"sub": authenticated_user.email})
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
async def sample_dataset(test_db, authenticated_user) -> Dataset:
    """Create sample dataset for testing."""
    dataset_data = {
        "name": "Test Dataset",
        "filename": "test_data.csv",
        "file_size": 1024,
        "file_path": "/tmp/test_data.csv",
        "columns": ["feature1", "feature2", "target"],
        "row_count": 100,
        "user_id": authenticated_user.id
    }
    dataset = Dataset(**dataset_data)
    await dataset.insert()
    return dataset


@pytest.fixture
async def sample_model(test_db, authenticated_user, sample_dataset) -> MLModel:
    """Create sample ML model for testing."""
    model_data = {
        "name": "Test Model",
        "model_type": "classification",
        "algorithm": "random_forest",
        "dataset_id": sample_dataset.id,
        "target_column": "target",
        "feature_columns": ["feature1", "feature2"],
        "model_path": "/tmp/test_model.pkl",
        "metrics": {"accuracy": 0.85, "f1_score": 0.82},
        "hyperparameters": {"n_estimators": 100, "max_depth": 10},
        "user_id": authenticated_user.id
    }
    model = MLModel(**model_data)
    await model.insert()
    return model


@pytest.fixture
async def sample_training_job(test_db, authenticated_user, sample_dataset) -> TrainingJob:
    """Create sample training job for testing."""
    job_data = {
        "dataset_id": sample_dataset.id,
        "model_type": "classification",
        "algorithm": "random_forest",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2"],
        "hyperparameters": {"n_estimators": 100, "max_depth": 10},
        "status": "completed",
        "user_id": authenticated_user.id
    }
    job = TrainingJob(**job_data)
    await job.insert()
    return job