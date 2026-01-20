"""
Test cases for data service functionality.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from backend.app.services.data_service import DataService


@pytest.mark.asyncio
class TestDataService:
    """Test data service operations."""
    
    def setup_method(self):
        """Set up test data service."""
        self.data_service = DataService()
        
        # Create sample dataframe for testing
        self.sample_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': ['A', 'B', 'A', 'B', 'A']
        })
    
    async def test_load_dataset_csv(self):
        """Test loading CSV dataset."""
        with patch('pandas.read_csv', return_value=self.sample_df):
            result = await self.data_service.load_dataset('/fake/path/data.csv', 'csv')
            
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ['feature1', 'feature2', 'target']
            assert len(result) == 5
    
    async def test_load_dataset_excel(self):
        """Test loading Excel dataset."""
        with patch('pandas.read_excel', return_value=self.sample_df):
            result = await self.data_service.load_dataset('/fake/path/data.xlsx', 'excel')
            
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ['feature1', 'feature2', 'target']
    
    async def test_load_dataset_json(self):
        """Test loading JSON dataset."""
        with patch('pandas.read_json', return_value=self.sample_df):
            result = await self.data_service.load_dataset('/fake/path/data.json', 'json')
            
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ['feature1', 'feature2', 'target']
    
    async def test_load_dataset_unsupported_format(self):
        """Test loading dataset with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            await self.data_service.load_dataset('/fake/path/data.txt', 'txt')
    
    async def test_analyze_dataset(self):
        """Test dataset analysis functionality."""
        analysis = await self.data_service.analyze_dataset(self.sample_df)
        
        assert 'column_info' in analysis
        assert 'statistics' in analysis
        assert 'missing_values' in analysis
        assert 'data_quality' in analysis
        
        # Check column info
        column_info = analysis['column_info']
        assert len(column_info) == 3
        assert column_info['feature1']['dtype'] == 'int64'
        assert column_info['target']['dtype'] == 'object'
        
        # Check statistics
        stats = analysis['statistics']
        assert 'feature1' in stats
        assert 'feature2' in stats
        assert stats['feature1']['mean'] == 3.0
        assert stats['feature1']['std'] > 0
    
    async def test_get_data_preview(self):
        """Test getting data preview."""
        preview = await self.data_service.get_data_preview(self.sample_df, rows=3)
        
        assert 'head' in preview
        assert 'shape' in preview
        assert 'columns' in preview
        
        assert len(preview['head']) == 3
        assert preview['shape'] == (5, 3)
        assert preview['columns'] == ['feature1', 'feature2', 'target']
    
    async def test_validate_dataset_for_ml(self):
        """Test ML dataset validation."""
        validation = await self.data_service.validate_dataset_for_ml(
            self.sample_df,
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )
        
        assert 'is_valid' in validation
        assert 'issues' in validation
        assert 'recommendations' in validation
        
        assert validation['is_valid'] is True
        assert len(validation['issues']) == 0
    
    async def test_validate_dataset_missing_target(self):
        """Test validation with missing target column."""
        validation = await self.data_service.validate_dataset_for_ml(
            self.sample_df,
            target_column='nonexistent_column',
            feature_columns=['feature1', 'feature2']
        )
        
        assert validation['is_valid'] is False
        assert any('Target column' in issue for issue in validation['issues'])
    
    async def test_validate_dataset_missing_features(self):
        """Test validation with missing feature columns."""
        validation = await self.data_service.validate_dataset_for_ml(
            self.sample_df,
            target_column='target',
            feature_columns=['feature1', 'nonexistent_feature']
        )
        
        assert validation['is_valid'] is False
        assert any('feature' in issue.lower() for issue in validation['issues'])
    
    async def test_prepare_data_for_training(self):
        """Test data preparation for training."""
        prepared = await self.data_service.prepare_data_for_training(
            self.sample_df,
            target_column='target',
            feature_columns=['feature1', 'feature2'],
            test_size=0.2,
            random_state=42
        )
        
        assert 'X_train' in prepared
        assert 'X_test' in prepared
        assert 'y_train' in prepared
        assert 'y_test' in prepared
        assert 'feature_names' in prepared
        
        # Check shapes
        assert len(prepared['X_train']) + len(prepared['X_test']) == 5
        assert prepared['X_train'].shape[1] == 2  # Two features
        assert prepared['feature_names'] == ['feature1', 'feature2']
    
    async def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create dataframe with missing values
        df_with_nan = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, 40, 50],
            'target': ['A', 'B', 'A', 'B', 'A']
        })
        
        cleaned_df = await self.data_service.handle_missing_values(
            df_with_nan,
            strategy='mean'
        )
        
        # Check that no missing values remain in numeric columns
        assert not cleaned_df['feature1'].isna().any()
        assert not cleaned_df['feature2'].isna().any()
    
    async def test_encode_categorical_variables(self):
        """Test categorical variable encoding."""
        encoded_df = await self.data_service.encode_categorical_variables(
            self.sample_df,
            categorical_columns=['target'],
            encoding_type='label'
        )
        
        # Check that target column is now numeric
        assert encoded_df['target'].dtype != 'object'
        assert all(isinstance(val, (int, float)) for val in encoded_df['target'])
    
    async def test_scale_features(self):
        """Test feature scaling."""
        scaled_data = await self.data_service.scale_features(
            self.sample_df[['feature1', 'feature2']],
            scaler_type='standard'
        )
        
        assert 'scaled_data' in scaled_data
        assert 'scaler' in scaled_data
        
        # Check that scaled data has similar shape
        assert scaled_data['scaled_data'].shape == (5, 2)
        
        # Check that scaling worked (mean should be close to 0 for standardization)
        scaled_df = pd.DataFrame(scaled_data['scaled_data'])
        assert abs(scaled_df.mean().iloc[0]) < 0.01  # Close to 0


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing."""
    with patch('aiofiles.open', create=True) as mock_open, \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'):
        yield mock_open