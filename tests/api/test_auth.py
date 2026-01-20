"""
Test cases for authentication API endpoints.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


class TestAuthAPI:
    """Test authentication endpoints."""
    
    def test_register_user_success(self, client: TestClient):
        """Test successful user registration."""
        user_data = {
            "email": "newuser@example.com",
            "password": "securepassword123"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"]
        assert "id" in data
        assert "password" not in data
    
    def test_register_user_duplicate_email(self, client: TestClient, authenticated_user):
        """Test registration with duplicate email fails."""
        user_data = {
            "email": authenticated_user.email,
            "password": "securepassword123"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    def test_register_user_invalid_email(self, client: TestClient):
        """Test registration with invalid email fails."""
        user_data = {
            "email": "invalid-email",
            "password": "securepassword123"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 422
    
    def test_login_success(self, client: TestClient, authenticated_user):
        """Test successful login."""
        login_data = {
            "username": authenticated_user.email,  # FastAPI OAuth2 uses 'username'
            "password": "testpass"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client: TestClient, authenticated_user):
        """Test login with invalid credentials fails."""
        login_data = {
            "username": authenticated_user.email,
            "password": "wrongpassword"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]
    
    def test_login_nonexistent_user(self, client: TestClient):
        """Test login with nonexistent user fails."""
        login_data = {
            "username": "nonexistent@example.com",
            "password": "password"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 401
    
    def test_get_current_user_success(self, client: TestClient, auth_headers, authenticated_user):
        """Test getting current user with valid token."""
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == authenticated_user.email
        assert data["id"] == str(authenticated_user.id)
    
    def test_get_current_user_no_token(self, client: TestClient):
        """Test getting current user without token fails."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 401
    
    def test_get_current_user_invalid_token(self, client: TestClient):
        """Test getting current user with invalid token fails."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == 401


@pytest.mark.asyncio
class TestAuthAPIAsync:
    """Test authentication endpoints with async client."""
    
    async def test_register_user_async(self, async_client: AsyncClient):
        """Test user registration with async client."""
        user_data = {
            "email": "asyncuser@example.com",
            "password": "securepassword123"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"]
    
    async def test_login_async(self, async_client: AsyncClient, authenticated_user):
        """Test login with async client."""
        login_data = {
            "username": authenticated_user.email,
            "password": "testpass"
        }
        
        response = await async_client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data