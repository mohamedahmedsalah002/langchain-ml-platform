"""Pydantic schemas for API request/response models."""
from datetime import datetime
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, EmailStr


# Authentication Schemas
class UserRegister(BaseModel):
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime
    is_active: bool


# Dataset Schemas
class DatasetResponse(BaseModel):
    id: str
    filename: str
    size: int
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    column_types: Optional[Dict[str, str]] = None
    column_names: Optional[List[str]] = None
    upload_date: datetime
    status: str


class DatasetProfile(BaseModel):
    dataset_id: str
    num_rows: int
    num_columns: int
    column_info: Dict[str, Any]
    missing_values: Dict[str, int]
    statistics: Dict[str, Any]


# Training Job Schemas
class TrainingJobCreate(BaseModel):
    dataset_id: str
    model_type: str
    problem_type: str
    target_column: str
    feature_columns: List[str]
    parameters: Dict = {}
    test_size: float = 0.2


class TrainingJobResponse(BaseModel):
    id: str
    dataset_id: str
    model_type: str
    problem_type: str
    target_column: str
    feature_columns: List[str]
    status: str
    progress: int
    current_stage: Optional[str] = None
    parameters: Dict
    metrics: Optional[Dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# Model Schemas
class MLModelResponse(BaseModel):
    id: str
    job_id: str
    model_type: str
    problem_type: str
    target_column: str
    feature_columns: List[str]
    metrics: Dict
    feature_importance: Optional[Dict] = None
    created_at: datetime
    version: int


class PredictionRequest(BaseModel):
    input_data: Dict


class PredictionResponse(BaseModel):
    prediction: Any
    probabilities: Optional[Dict] = None
    timestamp: datetime


# Chat Schemas
class ChatMessage(BaseModel):
    content: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime


# Health Check
class HealthCheck(BaseModel):
    status: str
    timestamp: datetime

