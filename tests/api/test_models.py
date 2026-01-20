"""
Test cases for ML models API endpoints.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


class TestModelsAPI:
    """Test ML models endpoints."""
    
    def test_get_models_success(self, client: TestClient, auth_headers, sample_model):
        """Test getting user models."""
        response = client.get("/api/v1/models/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(m["id"] == str(sample_model.id) for m in data)
    
    def test_get_models_unauthorized(self, client: TestClient):
        """Test getting models without authentication."""
        response = client.get("/api/v1/models/")
        
        assert response.status_code == 401
    
    def test_get_model_by_id_success(self, client: TestClient, auth_headers, sample_model):
        """Test getting specific model by ID."""
        response = client.get(f"/api/v1/models/{sample_model.id}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_model.id)
        assert data["name"] == sample_model.name
        assert data["model_type"] == sample_model.model_type
        assert data["algorithm"] == sample_model.algorithm
        assert "metrics" in data
    
    def test_get_model_by_id_not_found(self, client: TestClient, auth_headers):
        """Test getting non-existent model."""
        fake_id = "507f1f77bcf86cd799439011"
        response = client.get(f"/api/v1/models/{fake_id}", headers=auth_headers)
        
        assert response.status_code == 404
    
    def test_get_model_metrics_success(self, client: TestClient, auth_headers, sample_model):
        """Test getting model metrics."""
        response = client.get(
            f"/api/v1/models/{sample_model.id}/metrics",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "accuracy" in data
        assert "f1_score" in data
        assert isinstance(data["accuracy"], float)
    
    def test_get_model_feature_importance_success(self, client: TestClient, auth_headers, sample_model):
        """Test getting model feature importance."""
        response = client.get(
            f"/api/v1/models/{sample_model.id}/feature-importance",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Each item should have feature name and importance score
        if data:  # If feature importance data exists
            assert "feature" in data[0]
            assert "importance" in data[0]
    
    def test_make_prediction_success(self, client: TestClient, auth_headers, sample_model):
        """Test making predictions with trained model."""
        prediction_data = {
            "features": {
                "feature1": 1.5,
                "feature2": 2.3
            }
        }
        
        response = client.post(
            f"/api/v1/models/{sample_model.id}/predict",
            headers=auth_headers,
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "prediction_id" in data
    
    def test_make_prediction_invalid_features(self, client: TestClient, auth_headers, sample_model):
        """Test making predictions with invalid features."""
        prediction_data = {
            "features": {
                "invalid_feature": 1.5
            }
        }
        
        response = client.post(
            f"/api/v1/models/{sample_model.id}/predict",
            headers=auth_headers,
            json=prediction_data
        )
        
        assert response.status_code == 400
        assert "Invalid features" in response.json()["detail"]
    
    def test_make_prediction_model_not_found(self, client: TestClient, auth_headers):
        """Test making predictions with non-existent model."""
        fake_id = "507f1f77bcf86cd799439011"
        prediction_data = {
            "features": {
                "feature1": 1.5,
                "feature2": 2.3
            }
        }
        
        response = client.post(
            f"/api/v1/models/{fake_id}/predict",
            headers=auth_headers,
            json=prediction_data
        )
        
        assert response.status_code == 404
    
    def test_batch_prediction_success(self, client: TestClient, auth_headers, sample_model):
        """Test batch predictions."""
        batch_data = {
            "predictions": [
                {"feature1": 1.5, "feature2": 2.3},
                {"feature1": 0.8, "feature2": 1.2}
            ]
        }
        
        response = client.post(
            f"/api/v1/models/{sample_model.id}/batch-predict",
            headers=auth_headers,
            json=batch_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert all("prediction" in p for p in data["predictions"])
    
    def test_delete_model_success(self, client: TestClient, auth_headers, sample_model):
        """Test successful model deletion."""
        response = client.delete(
            f"/api/v1/models/{sample_model.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        # Verify model is deleted
        get_response = client.get(
            f"/api/v1/models/{sample_model.id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404
    
    def test_delete_model_not_found(self, client: TestClient, auth_headers):
        """Test deleting non-existent model."""
        fake_id = "507f1f77bcf86cd799439011"
        response = client.delete(
            f"/api/v1/models/{fake_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 404


@pytest.mark.asyncio
class TestModelsAPIAsync:
    """Test models endpoints with async client."""
    
    async def test_get_models_async(self, async_client: AsyncClient, auth_headers, sample_model):
        """Test getting models with async client."""
        response = await async_client.get("/api/v1/models/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    async def test_make_prediction_async(self, async_client: AsyncClient, auth_headers, sample_model):
        """Test making predictions with async client."""
        prediction_data = {
            "features": {
                "feature1": 1.5,
                "feature2": 2.3
            }
        }
        
        response = await async_client.post(
            f"/api/v1/models/{sample_model.id}/predict",
            headers=auth_headers,
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data