"""
Test cases for User model.
"""
import pytest
from beanie.exceptions import ValidationError
from backend.app.models.user import User


@pytest.mark.asyncio
class TestUserModel:
    """Test User database model."""
    
    async def test_create_user_success(self, test_db):
        """Test creating a valid user."""
        user_data = {
            "email": "test@example.com",
            "hashed_password": "hashed_password_123",
            "is_active": True
        }
        
        user = User(**user_data)
        await user.insert()
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.hashed_password == "hashed_password_123"
        assert user.is_active is True
        assert user.created_at is not None
    
    async def test_create_user_invalid_email(self, test_db):
        """Test creating user with invalid email format."""
        user_data = {
            "email": "invalid-email",
            "hashed_password": "hashed_password_123"
        }
        
        with pytest.raises(ValidationError):
            user = User(**user_data)
            await user.insert()
    
    async def test_user_email_unique(self, test_db):
        """Test that user email must be unique."""
        user_data = {
            "email": "duplicate@example.com",
            "hashed_password": "hashed_password_123"
        }
        
        # Create first user
        user1 = User(**user_data)
        await user1.insert()
        
        # Try to create second user with same email
        user2 = User(**user_data)
        
        with pytest.raises(Exception):  # Should raise duplicate key error
            await user2.insert()
    
    async def test_find_user_by_email(self, test_db):
        """Test finding user by email."""
        user_data = {
            "email": "findme@example.com",
            "hashed_password": "hashed_password_123"
        }
        
        user = User(**user_data)
        await user.insert()
        
        # Find user by email
        found_user = await User.find_one(User.email == "findme@example.com")
        
        assert found_user is not None
        assert found_user.email == "findme@example.com"
        assert found_user.id == user.id
    
    async def test_update_user(self, test_db):
        """Test updating user information."""
        user_data = {
            "email": "update@example.com",
            "hashed_password": "hashed_password_123"
        }
        
        user = User(**user_data)
        await user.insert()
        
        # Update user
        user.is_active = False
        await user.save()
        
        # Verify update
        updated_user = await User.get(user.id)
        assert updated_user.is_active is False
    
    async def test_delete_user(self, test_db):
        """Test deleting user."""
        user_data = {
            "email": "delete@example.com", 
            "hashed_password": "hashed_password_123"
        }
        
        user = User(**user_data)
        await user.insert()
        user_id = user.id
        
        # Delete user
        await user.delete()
        
        # Verify deletion
        deleted_user = await User.get(user_id)
        assert deleted_user is None
    
    async def test_user_default_values(self, test_db):
        """Test user model default values."""
        user_data = {
            "email": "defaults@example.com",
            "hashed_password": "hashed_password_123"
        }
        
        user = User(**user_data)
        await user.insert()
        
        # Check default values
        assert user.is_active is True  # Default should be True
        assert user.created_at is not None