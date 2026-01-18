"""Chat/LangChain API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from bson import ObjectId
from typing import Optional

from app.api.schemas import ChatMessage, ChatResponse
from app.models.user import User
from app.models.chat_session import ChatSession
from app.utils.auth import get_current_user
from app.services.langchain_service import LangChainService


router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("/message", response_model=ChatResponse)
async def send_message(
    message: ChatMessage,
    current_user: User = Depends(get_current_user)
):
    """Send a message to the AI assistant."""
    # Get or create chat session
    if message.session_id:
        session = await ChatSession.get(ObjectId(message.session_id))
        if not session or session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        session = ChatSession(
            user_id=current_user.id,
            messages=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        await session.insert()
    
    # Add user message to session
    user_message = {
        "role": "user",
        "content": message.content,
        "timestamp": datetime.utcnow().isoformat()
    }
    session.messages.append(user_message)
    
    # Process with LangChain
    langchain_service = LangChainService()
    response_text = await langchain_service.chat(message.content)
    
    # Add assistant response to session
    assistant_message = {
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.utcnow().isoformat()
    }
    session.messages.append(assistant_message)
    session.updated_at = datetime.utcnow()
    
    await session.save()
    
    return ChatResponse(
        response=response_text,
        session_id=str(session.id),
        timestamp=datetime.utcnow()
    )


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get chat history for a session."""
    session = await ChatSession.get(ObjectId(session_id))
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return {
        "session_id": str(session.id),
        "messages": session.messages,
        "created_at": session.created_at,
        "updated_at": session.updated_at
    }


@router.get("/sessions")
async def list_chat_sessions(current_user: User = Depends(get_current_user)):
    """List all chat sessions for the current user."""
    sessions = await ChatSession.find(
        ChatSession.user_id == current_user.id
    ).sort(-ChatSession.updated_at).to_list()
    
    return [
        {
            "session_id": str(session.id),
            "message_count": len(session.messages),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "last_message": session.messages[-1]["content"][:100] if session.messages else None
        }
        for session in sessions
    ]

