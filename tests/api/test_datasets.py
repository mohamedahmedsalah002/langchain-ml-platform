"""
Test cases for dataset API endpoints.
"""
import pytest
import io
from httpx import AsyncClient
from fastapi.testclient import TestClient


class TestDatasetAPI:
    """Test dataset endpoints."""
    
    def test_get_datasets_success(self, client: TestClient, auth_headers, sample_dataset):
        """Test getting user datasets."""
        response = client.get("/api/v1/datasets/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(d["id"] == str(sample_dataset.id) for d in data)
    
    def test_get_datasets_unauthorized(self, client: TestClient):
        """Test getting datasets without authentication."""
        response = client.get("/api/v1/datasets/")
        
        assert response.status_code == 401
    
    def test_get_dataset_by_id_success(self, client: TestClient, auth_headers, sample_dataset):
        """Test getting specific dataset by ID."""
        response = client.get(f"/api/v1/datasets/{sample_dataset.id}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_dataset.id)
        assert data["name"] == sample_dataset.name
        assert data["columns"] == sample_dataset.columns
    
    def test_get_dataset_by_id_not_found(self, client: TestClient, auth_headers):
        """Test getting non-existent dataset."""
        fake_id = "507f1f77bcf86cd799439011"  # Valid ObjectId format
        response = client.get(f"/api/v1/datasets/{fake_id}", headers=auth_headers)
        
        assert response.status_code == 404
    
    def test_get_dataset_by_id_invalid_format(self, client: TestClient, auth_headers):
        """Test getting dataset with invalid ID format."""
        response = client.get("/api/v1/datasets/invalid_id", headers=auth_headers)
        
        assert response.status_code == 422
    
    def test_upload_dataset_csv_success(self, client: TestClient, auth_headers):
        """Test successful CSV dataset upload."""
        # Create a simple CSV content
        csv_content = "feature1,feature2,target\n1,2,A\n3,4,B\n5,6,A"
        
        files = {
            "file": ("test_dataset.csv", io.StringIO(csv_content), "text/csv")
        }
        data = {
            "name": "Test CSV Dataset"
        }
        
        response = client.post(
            "/api/v1/datasets/upload",
            headers=auth_headers,
            files=files,
            data=data
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["name"] == "Test CSV Dataset"
        assert result["filename"] == "test_dataset.csv"
        assert "columns" in result
        assert result["row_count"] > 0
    
    def test_upload_dataset_no_file(self, client: TestClient, auth_headers):
        """Test dataset upload without file."""
        data = {"name": "Test Dataset"}
        
        response = client.post(
            "/api/v1/datasets/upload",
            headers=auth_headers,
            data=data
        )
        
        assert response.status_code == 422
    
    def test_upload_dataset_invalid_format(self, client: TestClient, auth_headers):
        """Test upload with unsupported file format."""
        files = {
            "file": ("test.txt", io.StringIO("invalid content"), "text/plain")
        }
        data = {"name": "Test Dataset"}
        
        response = client.post(
            "/api/v1/datasets/upload",
            headers=auth_headers,
            files=files,
            data=data
        )
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    
    def test_get_dataset_preview_success(self, client: TestClient, auth_headers, sample_dataset):
        """Test getting dataset preview."""
        response = client.get(
            f"/api/v1/datasets/{sample_dataset.id}/preview",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "columns" in data
        assert "sample_data" in data
        assert "statistics" in data
    
    def test_get_dataset_preview_not_found(self, client: TestClient, auth_headers):
        """Test getting preview for non-existent dataset."""
        fake_id = "507f1f77bcf86cd799439011"
        response = client.get(
            f"/api/v1/datasets/{fake_id}/preview",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_delete_dataset_success(self, client: TestClient, auth_headers, sample_dataset):
        """Test successful dataset deletion."""
        response = client.delete(
            f"/api/v1/datasets/{sample_dataset.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        # Verify dataset is deleted
        get_response = client.get(
            f"/api/v1/datasets/{sample_dataset.id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404
    
    def test_delete_dataset_not_found(self, client: TestClient, auth_headers):
        """Test deleting non-existent dataset."""
        fake_id = "507f1f77bcf86cd799439011"
        response = client.delete(
            f"/api/v1/datasets/{fake_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 404


@pytest.mark.asyncio
class TestDatasetAPIAsync:
    """Test dataset endpoints with async client."""
    
    async def test_get_datasets_async(self, async_client: AsyncClient, auth_headers, sample_dataset):
        """Test getting datasets with async client."""
        response = await async_client.get("/api/v1/datasets/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    async def test_upload_dataset_async(self, async_client: AsyncClient, auth_headers):
        """Test dataset upload with async client."""
        csv_content = "x,y,label\n1,2,0\n3,4,1"
        
        files = {
            "file": ("async_test.csv", csv_content, "text/csv")
        }
        data = {"name": "Async Test Dataset"}
        
        response = await async_client.post(
            "/api/v1/datasets/upload",
            headers=auth_headers,
            files=files,
            data=data
        )
        
        assert response.status_code == 201