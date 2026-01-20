"""
Test cases for training API endpoints.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


class TestTrainingAPI:
    """Test training endpoints."""
    
    def test_get_training_jobs_success(self, client: TestClient, auth_headers, sample_training_job):
        """Test getting user training jobs."""
        response = client.get("/api/v1/training/jobs", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(j["id"] == str(sample_training_job.id) for j in data)
    
    def test_get_training_jobs_unauthorized(self, client: TestClient):
        """Test getting training jobs without authentication."""
        response = client.get("/api/v1/training/jobs")
        
        assert response.status_code == 401
    
    def test_get_training_job_by_id_success(self, client: TestClient, auth_headers, sample_training_job):
        """Test getting specific training job by ID."""
        response = client.get(
            f"/api/v1/training/jobs/{sample_training_job.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_training_job.id)
        assert data["status"] == sample_training_job.status
        assert data["algorithm"] == sample_training_job.algorithm
    
    def test_get_training_job_by_id_not_found(self, client: TestClient, auth_headers):
        """Test getting non-existent training job."""
        fake_id = "507f1f77bcf86cd799439011"
        response = client.get(f"/api/v1/training/jobs/{fake_id}", headers=auth_headers)
        
        assert response.status_code == 404
    
    def test_start_training_success(self, client: TestClient, auth_headers, sample_dataset):
        """Test starting new training job."""
        training_config = {
            "dataset_id": str(sample_dataset.id),
            "model_name": "Test Classification Model",
            "model_type": "classification",
            "algorithm": "random_forest",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"],
            "hyperparameters": {
                "n_estimators": 50,
                "max_depth": 5,
                "random_state": 42
            },
            "validation_split": 0.2
        }
        
        response = client.post(
            "/api/v1/training/start",
            headers=auth_headers,
            json=training_config
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending"
        assert data["algorithm"] == "random_forest"
        assert "id" in data
        assert "created_at" in data
    
    def test_start_training_invalid_dataset(self, client: TestClient, auth_headers):
        """Test starting training with non-existent dataset."""
        fake_dataset_id = "507f1f77bcf86cd799439011"
        training_config = {
            "dataset_id": fake_dataset_id,
            "model_name": "Test Model",
            "model_type": "classification",
            "algorithm": "random_forest",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"]
        }
        
        response = client.post(
            "/api/v1/training/start",
            headers=auth_headers,
            json=training_config
        )
        
        assert response.status_code == 404
        assert "Dataset not found" in response.json()["detail"]
    
    def test_start_training_invalid_algorithm(self, client: TestClient, auth_headers, sample_dataset):
        """Test starting training with invalid algorithm."""
        training_config = {
            "dataset_id": str(sample_dataset.id),
            "model_name": "Test Model",
            "model_type": "classification",
            "algorithm": "invalid_algorithm",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"]
        }
        
        response = client.post(
            "/api/v1/training/start",
            headers=auth_headers,
            json=training_config
        )
        
        assert response.status_code == 400
        assert "Unsupported algorithm" in response.json()["detail"]
    
    def test_start_training_missing_target_column(self, client: TestClient, auth_headers, sample_dataset):
        """Test starting training without target column."""
        training_config = {
            "dataset_id": str(sample_dataset.id),
            "model_name": "Test Model",
            "model_type": "classification",
            "algorithm": "random_forest",
            "target_column": "nonexistent_column",
            "feature_columns": ["feature1", "feature2"]
        }
        
        response = client.post(
            "/api/v1/training/start",
            headers=auth_headers,
            json=training_config
        )
        
        assert response.status_code == 400
        assert "Target column not found" in response.json()["detail"]
    
    def test_cancel_training_success(self, client: TestClient, auth_headers, sample_training_job):
        """Test canceling training job."""
        response = client.post(
            f"/api/v1/training/jobs/{sample_training_job.id}/cancel",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"
    
    def test_cancel_training_not_found(self, client: TestClient, auth_headers):
        """Test canceling non-existent training job."""
        fake_id = "507f1f77bcf86cd799439011"
        response = client.post(
            f"/api/v1/training/jobs/{fake_id}/cancel",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_get_training_logs_success(self, client: TestClient, auth_headers, sample_training_job):
        """Test getting training job logs."""
        response = client.get(
            f"/api/v1/training/jobs/{sample_training_job.id}/logs",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)
    
    def test_get_training_progress_success(self, client: TestClient, auth_headers, sample_training_job):
        """Test getting training job progress."""
        response = client.get(
            f"/api/v1/training/jobs/{sample_training_job.id}/progress",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "progress_percentage" in data
        assert "current_step" in data
        assert "total_steps" in data
    
    def test_get_available_algorithms_success(self, client: TestClient, auth_headers):
        """Test getting available ML algorithms."""
        response = client.get("/api/v1/training/algorithms", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert "regression" in data
        assert isinstance(data["classification"], list)
        assert isinstance(data["regression"], list)
        
        # Check that common algorithms are included
        classification_algorithms = data["classification"]
        assert any("random_forest" in alg["name"] for alg in classification_algorithms)
        assert any("logistic_regression" in alg["name"] for alg in classification_algorithms)


@pytest.mark.asyncio
class TestTrainingAPIAsync:
    """Test training endpoints with async client."""
    
    async def test_get_training_jobs_async(self, async_client: AsyncClient, auth_headers, sample_training_job):
        """Test getting training jobs with async client."""
        response = await async_client.get("/api/v1/training/jobs", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    async def test_start_training_async(self, async_client: AsyncClient, auth_headers, sample_dataset):
        """Test starting training with async client."""
        training_config = {
            "dataset_id": str(sample_dataset.id),
            "model_name": "Async Test Model",
            "model_type": "classification",
            "algorithm": "random_forest",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"]
        }
        
        response = await async_client.post(
            "/api/v1/training/start",
            headers=auth_headers,
            json=training_config
        )
        
        assert response.status_code == 201