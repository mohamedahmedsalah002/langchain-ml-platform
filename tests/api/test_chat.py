"""
Test cases for chat/AI assistant API endpoints.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


class TestChatAPI:
    """Test chat/AI assistant endpoints."""
    
    def test_get_chat_sessions_success(self, client: TestClient, auth_headers):
        """Test getting user chat sessions."""
        response = client.get("/api/v1/chat/sessions", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_chat_sessions_unauthorized(self, client: TestClient):
        """Test getting chat sessions without authentication."""
        response = client.get("/api/v1/chat/sessions")
        
        assert response.status_code == 401
    
    def test_create_chat_session_success(self, client: TestClient, auth_headers):
        """Test creating new chat session."""
        session_data = {
            "title": "Test Chat Session"
        }
        
        response = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json=session_data
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Chat Session"
        assert "id" in data
        assert "created_at" in data
        assert data["messages"] == []
    
    def test_send_message_success(self, client: TestClient, auth_headers):
        """Test sending message to AI assistant."""
        # First create a chat session
        session_response = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json={"title": "Test Session"}
        )
        session_id = session_response.json()["id"]
        
        # Send a message
        message_data = {
            "message": "What is machine learning?",
            "context": {
                "dataset_id": None,
                "model_id": None
            }
        }
        
        response = client.post(
            f"/api/v1/chat/sessions/{session_id}/message",
            headers=auth_headers,
            json=message_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "message_id" in data
        assert isinstance(data["response"], str)
    
    def test_send_message_with_dataset_context(self, client: TestClient, auth_headers, sample_dataset):
        """Test sending message with dataset context."""
        # Create chat session
        session_response = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json={"title": "Dataset Chat"}
        )
        session_id = session_response.json()["id"]
        
        # Send message with dataset context
        message_data = {
            "message": "Analyze my dataset",
            "context": {
                "dataset_id": str(sample_dataset.id),
                "model_id": None
            }
        }
        
        response = client.post(
            f"/api/v1/chat/sessions/{session_id}/message",
            headers=auth_headers,
            json=message_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        # Response should mention the dataset
        assert sample_dataset.name.lower() in data["response"].lower()
    
    def test_send_message_with_model_context(self, client: TestClient, auth_headers, sample_model):
        """Test sending message with model context."""
        # Create chat session
        session_response = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json={"title": "Model Chat"}
        )
        session_id = session_response.json()["id"]
        
        # Send message with model context
        message_data = {
            "message": "Explain my model performance",
            "context": {
                "dataset_id": None,
                "model_id": str(sample_model.id)
            }
        }
        
        response = client.post(
            f"/api/v1/chat/sessions/{session_id}/message",
            headers=auth_headers,
            json=message_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
    
    def test_send_message_session_not_found(self, client: TestClient, auth_headers):
        """Test sending message to non-existent session."""
        fake_id = "507f1f77bcf86cd799439011"
        message_data = {
            "message": "Test message",
            "context": {}
        }
        
        response = client.post(
            f"/api/v1/chat/sessions/{fake_id}/message",
            headers=auth_headers,
            json=message_data
        )
        
        assert response.status_code == 404
    
    def test_get_chat_session_by_id_success(self, client: TestClient, auth_headers):
        """Test getting specific chat session."""
        # Create session
        session_response = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json={"title": "Test Session"}
        )
        session_id = session_response.json()["id"]
        
        # Get session by ID
        response = client.get(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert data["title"] == "Test Session"
        assert "messages" in data
    
    def test_get_chat_session_by_id_not_found(self, client: TestClient, auth_headers):
        """Test getting non-existent chat session."""
        fake_id = "507f1f77bcf86cd799439011"
        response = client.get(f"/api/v1/chat/sessions/{fake_id}", headers=auth_headers)
        
        assert response.status_code == 404
    
    def test_delete_chat_session_success(self, client: TestClient, auth_headers):
        """Test deleting chat session."""
        # Create session
        session_response = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json={"title": "To Delete"}
        )
        session_id = session_response.json()["id"]
        
        # Delete session
        response = client.delete(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        # Verify deletion
        get_response = client.get(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404
    
    def test_get_quick_actions_success(self, client: TestClient, auth_headers):
        """Test getting available quick actions."""
        response = client.get("/api/v1/chat/quick-actions", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should contain common quick actions
        action_names = [action["name"] for action in data]
        assert any("dataset" in name.lower() for name in action_names)
        assert any("model" in name.lower() for name in action_names)


@pytest.mark.asyncio
class TestChatAPIAsync:
    """Test chat endpoints with async client."""
    
    async def test_create_chat_session_async(self, async_client: AsyncClient, auth_headers):
        """Test creating chat session with async client."""
        session_data = {"title": "Async Test Session"}
        
        response = await async_client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json=session_data
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Async Test Session"
    
    async def test_send_message_async(self, async_client: AsyncClient, auth_headers):
        """Test sending message with async client."""
        # Create session first
        session_response = await async_client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json={"title": "Async Chat"}
        )
        session_id = session_response.json()["id"]
        
        # Send message
        message_data = {
            "message": "Hello AI assistant",
            "context": {}
        }
        
        response = await async_client.post(
            f"/api/v1/chat/sessions/{session_id}/message",
            headers=auth_headers,
            json=message_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data