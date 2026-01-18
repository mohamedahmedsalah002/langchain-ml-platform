"""User database model."""
from datetime import datetime
from typing import Optional
from beanie import Document, Indexed
from pydantic import EmailStr


class User(Document):
    """User document model."""
    
    email: Indexed(EmailStr, unique=True)
    password_hash: str
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    is_active: bool = True
    
    class Settings:
        name = "users"
        indexes = [
            "email",
            "created_at",
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password_hash": "hashed_password",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "is_active": True,
            }
        }

