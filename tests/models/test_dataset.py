"""
Test cases for Dataset model.
"""
import pytest
from datetime import datetime
from beanie.exceptions import ValidationError
from backend.app.models.dataset import Dataset


@pytest.mark.asyncio
class TestDatasetModel:
    """Test Dataset database model."""
    
    async def test_create_dataset_success(self, test_db, authenticated_user):
        """Test creating a valid dataset."""
        dataset_data = {
            "name": "Test Dataset",
            "filename": "test_data.csv",
            "file_size": 1024,
            "file_path": "/path/to/test_data.csv",
            "columns": ["col1", "col2", "target"],
            "row_count": 100,
            "user_id": authenticated_user.id
        }
        
        dataset = Dataset(**dataset_data)
        await dataset.insert()
        
        assert dataset.id is not None
        assert dataset.name == "Test Dataset"
        assert dataset.filename == "test_data.csv"
        assert dataset.file_size == 1024
        assert dataset.columns == ["col1", "col2", "target"]
        assert dataset.row_count == 100
        assert dataset.user_id == authenticated_user.id
        assert dataset.created_at is not None
    
    async def test_create_dataset_missing_required_fields(self, test_db, authenticated_user):
        """Test creating dataset without required fields."""
        dataset_data = {
            "filename": "test.csv",
            # Missing name field
            "user_id": authenticated_user.id
        }
        
        with pytest.raises(ValidationError):
            dataset = Dataset(**dataset_data)
            await dataset.insert()
    
    async def test_dataset_column_types(self, test_db, authenticated_user):
        """Test dataset with column type information."""
        dataset_data = {
            "name": "Typed Dataset",
            "filename": "typed_data.csv",
            "file_size": 2048,
            "file_path": "/path/to/typed_data.csv",
            "columns": ["numeric_col", "text_col", "target"],
            "column_types": {
                "numeric_col": "float64",
                "text_col": "object", 
                "target": "int64"
            },
            "row_count": 200,
            "user_id": authenticated_user.id
        }
        
        dataset = Dataset(**dataset_data)
        await dataset.insert()
        
        assert dataset.column_types is not None
        assert dataset.column_types["numeric_col"] == "float64"
        assert dataset.column_types["text_col"] == "object"
    
    async def test_dataset_statistics(self, test_db, authenticated_user):
        """Test dataset with statistical information."""
        dataset_data = {
            "name": "Stats Dataset",
            "filename": "stats_data.csv",
            "file_size": 1024,
            "file_path": "/path/to/stats_data.csv",
            "columns": ["feature1", "feature2"],
            "row_count": 150,
            "statistics": {
                "feature1": {
                    "mean": 5.5,
                    "std": 2.1,
                    "min": 1.0,
                    "max": 10.0
                },
                "feature2": {
                    "mean": 0.3,
                    "std": 0.1,
                    "min": 0.0,
                    "max": 1.0
                }
            },
            "user_id": authenticated_user.id
        }
        
        dataset = Dataset(**dataset_data)
        await dataset.insert()
        
        assert dataset.statistics is not None
        assert dataset.statistics["feature1"]["mean"] == 5.5
        assert dataset.statistics["feature2"]["std"] == 0.1
    
    async def test_find_datasets_by_user(self, test_db, authenticated_user):
        """Test finding datasets by user ID."""
        # Create multiple datasets for the user
        dataset1_data = {
            "name": "Dataset 1",
            "filename": "data1.csv",
            "file_size": 1024,
            "file_path": "/path/to/data1.csv",
            "columns": ["col1"],
            "row_count": 50,
            "user_id": authenticated_user.id
        }
        
        dataset2_data = {
            "name": "Dataset 2", 
            "filename": "data2.csv",
            "file_size": 2048,
            "file_path": "/path/to/data2.csv",
            "columns": ["col2"],
            "row_count": 100,
            "user_id": authenticated_user.id
        }
        
        dataset1 = Dataset(**dataset1_data)
        dataset2 = Dataset(**dataset2_data)
        await dataset1.insert()
        await dataset2.insert()
        
        # Find all datasets for user
        user_datasets = await Dataset.find(Dataset.user_id == authenticated_user.id).to_list()
        
        assert len(user_datasets) >= 2
        dataset_names = [d.name for d in user_datasets]
        assert "Dataset 1" in dataset_names
        assert "Dataset 2" in dataset_names
    
    async def test_update_dataset(self, test_db, authenticated_user):
        """Test updating dataset information."""
        dataset_data = {
            "name": "Original Name",
            "filename": "original.csv",
            "file_size": 1024,
            "file_path": "/path/to/original.csv",
            "columns": ["col1"],
            "row_count": 50,
            "user_id": authenticated_user.id
        }
        
        dataset = Dataset(**dataset_data)
        await dataset.insert()
        
        # Update dataset
        dataset.name = "Updated Name"
        dataset.row_count = 75
        await dataset.save()
        
        # Verify update
        updated_dataset = await Dataset.get(dataset.id)
        assert updated_dataset.name == "Updated Name"
        assert updated_dataset.row_count == 75
    
    async def test_delete_dataset(self, test_db, authenticated_user):
        """Test deleting dataset."""
        dataset_data = {
            "name": "To Delete",
            "filename": "delete.csv",
            "file_size": 1024,
            "file_path": "/path/to/delete.csv",
            "columns": ["col1"],
            "row_count": 50,
            "user_id": authenticated_user.id
        }
        
        dataset = Dataset(**dataset_data)
        await dataset.insert()
        dataset_id = dataset.id
        
        # Delete dataset
        await dataset.delete()
        
        # Verify deletion
        deleted_dataset = await Dataset.get(dataset_id)
        assert deleted_dataset is None