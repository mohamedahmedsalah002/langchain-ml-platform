"""Chat session database model for LangChain conversations."""
from datetime import datetime
from typing import List, Dict
from beanie import Document, Indexed, PydanticObjectId


class ChatSession(Document):
    """Chat session document model."""
    
    user_id: Indexed(PydanticObjectId)
    messages: List[Dict] = []  # {role: str, content: str, timestamp: datetime}
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    
    class Settings:
        name = "chat_sessions"
        indexes = [
            "user_id",
            "created_at",
            "updated_at",
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "messages": [
                    {
                        "role": "user",
                        "content": "Analyze my dataset",
                        "timestamp": "2024-01-01T00:00:00"
                    },
                    {
                        "role": "assistant",
                        "content": "I'll analyze your dataset...",
                        "timestamp": "2024-01-01T00:00:05"
                    }
                ],
                "created_at": "2024-01-01T00:00:00",
            }
        }

