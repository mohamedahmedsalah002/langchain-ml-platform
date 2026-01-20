"""
Test cases for model training Celery tasks.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from backend.app.tasks.train_model import train_model_task


@pytest.mark.asyncio
class TestTrainModelTask:
    """Test model training background tasks."""
    
    def setup_method(self):
        """Set up test data for model training."""
        # Sample training data
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        self.training_params = {
            'job_id': 'test_job_123',
            'dataset_path': '/fake/path/data.csv',
            'model_name': 'Test Model',
            'model_type': 'classification',
            'algorithm': 'random_forest',
            'target_column': 'target',
            'feature_columns': ['feature1', 'feature2'],
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            },
            'validation_split': 0.2
        }
    
    @patch('backend.app.tasks.train_model.pd.read_csv')
    @patch('backend.app.tasks.train_model.joblib.dump')
    async def test_train_random_forest_classification(self, mock_joblib_dump, mock_read_csv):
        """Test training random forest classification model."""
        mock_read_csv.return_value = self.sample_data
        mock_joblib_dump.return_value = None
        
        with patch('backend.app.models.training_job.TrainingJob') as MockTrainingJob, \
             patch('backend.app.models.ml_model.MLModel') as MockMLModel:
            
            # Mock database operations
            mock_job = Mock()
            mock_job.id = 'test_job_123'
            mock_job.save = AsyncMock()
            MockTrainingJob.get = AsyncMock(return_value=mock_job)
            
            mock_model = Mock()
            mock_model.insert = AsyncMock()
            MockMLModel.return_value = mock_model
            
            # Run the training task
            result = await train_model_task(self.training_params)
            
            assert result['status'] == 'completed'
            assert result['model_id'] is not None
            assert 'metrics' in result
            assert 'accuracy' in result['metrics']
    
    @patch('backend.app.tasks.train_model.pd.read_csv')
    async def test_train_logistic_regression(self, mock_read_csv):
        """Test training logistic regression model."""
        mock_read_csv.return_value = self.sample_data
        
        params = self.training_params.copy()
        params['algorithm'] = 'logistic_regression'
        params['hyperparameters'] = {
            'C': 1.0,
            'random_state': 42
        }
        
        with patch('backend.app.models.training_job.TrainingJob') as MockTrainingJob, \
             patch('backend.app.models.ml_model.MLModel') as MockMLModel, \
             patch('joblib.dump'):
            
            mock_job = Mock()
            mock_job.save = AsyncMock()
            MockTrainingJob.get = AsyncMock(return_value=mock_job)
            
            mock_model = Mock()
            mock_model.insert = AsyncMock()
            MockMLModel.return_value = mock_model
            
            result = await train_model_task(params)
            
            assert result['status'] == 'completed'
            assert result['model_id'] is not None
    
    @patch('backend.app.tasks.train_model.pd.read_csv')
    async def test_train_regression_model(self, mock_read_csv):
        """Test training regression model."""
        # Modify data for regression
        regression_data = self.sample_data.copy()
        regression_data['target'] = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1]
        mock_read_csv.return_value = regression_data
        
        params = self.training_params.copy()
        params['model_type'] = 'regression'
        params['algorithm'] = 'linear_regression'
        params['hyperparameters'] = {}
        
        with patch('backend.app.models.training_job.TrainingJob') as MockTrainingJob, \
             patch('backend.app.models.ml_model.MLModel') as MockMLModel, \
             patch('joblib.dump'):
            
            mock_job = Mock()
            mock_job.save = AsyncMock()
            MockTrainingJob.get = AsyncMock(return_value=mock_job)
            
            mock_model = Mock()
            mock_model.insert = AsyncMock()
            MockMLModel.return_value = mock_model
            
            result = await train_model_task(params)
            
            assert result['status'] == 'completed'
            assert 'r2_score' in result['metrics']
    
    @patch('backend.app.tasks.train_model.pd.read_csv')
    async def test_invalid_algorithm(self, mock_read_csv):
        """Test training with invalid algorithm."""
        mock_read_csv.return_value = self.sample_data
        
        params = self.training_params.copy()
        params['algorithm'] = 'invalid_algorithm'
        
        with patch('backend.app.models.training_job.TrainingJob') as MockTrainingJob:
            mock_job = Mock()
            mock_job.save = AsyncMock()
            MockTrainingJob.get = AsyncMock(return_value=mock_job)
            
            result = await train_model_task(params)
            
            assert result['status'] == 'failed'
            assert 'error' in result
            assert 'Unsupported algorithm' in result['error']
    
    @patch('backend.app.tasks.train_model.pd.read_csv')
    async def test_missing_target_column(self, mock_read_csv):
        """Test training with missing target column."""
        mock_read_csv.return_value = self.sample_data
        
        params = self.training_params.copy()
        params['target_column'] = 'nonexistent_column'
        
        with patch('backend.app.models.training_job.TrainingJob') as MockTrainingJob:
            mock_job = Mock()
            mock_job.save = AsyncMock()
            MockTrainingJob.get = AsyncMock(return_value=mock_job)
            
            result = await train_model_task(params)
            
            assert result['status'] == 'failed'
            assert 'error' in result
            assert 'Target column' in result['error']
    
    @patch('backend.app.tasks.train_model.pd.read_csv')
    async def test_missing_feature_columns(self, mock_read_csv):
        """Test training with missing feature columns."""
        mock_read_csv.return_value = self.sample_data
        
        params = self.training_params.copy()
        params['feature_columns'] = ['nonexistent_feature']
        
        with patch('backend.app.models.training_job.TrainingJob') as MockTrainingJob:
            mock_job = Mock()
            mock_job.save = AsyncMock()
            MockTrainingJob.get = AsyncMock(return_value=mock_job)
            
            result = await train_model_task(params)
            
            assert result['status'] == 'failed'
            assert 'error' in result
    
    @patch('backend.app.tasks.train_model.pd.read_csv')
    async def test_data_preprocessing(self, mock_read_csv):
        """Test data preprocessing during training."""
        # Data with missing values and categorical features
        messy_data = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],
            'feature2': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        mock_read_csv.return_value = messy_data
        
        params = self.training_params.copy()
        params['feature_columns'] = ['feature1', 'feature2']
        
        with patch('backend.app.models.training_job.TrainingJob') as MockTrainingJob, \
             patch('backend.app.models.ml_model.MLModel') as MockMLModel, \
             patch('joblib.dump'):
            
            mock_job = Mock()
            mock_job.save = AsyncMock()
            MockTrainingJob.get = AsyncMock(return_value=mock_job)
            
            mock_model = Mock()
            mock_model.insert = AsyncMock()
            MockMLModel.return_value = mock_model
            
            result = await train_model_task(params)
            
            # Should handle preprocessing successfully
            assert result['status'] == 'completed'
    
    async def test_hyperparameter_validation(self):
        """Test hyperparameter validation."""
        from backend.app.tasks.train_model import validate_hyperparameters
        
        # Valid hyperparameters for random forest
        valid_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42
        }
        
        result = validate_hyperparameters('random_forest', valid_params)
        assert result['is_valid'] is True
        
        # Invalid hyperparameters
        invalid_params = {
            'n_estimators': -5,  # Invalid negative value
            'unknown_param': 'value'  # Unknown parameter
        }
        
        result = validate_hyperparameters('random_forest', invalid_params)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
    
    async def test_model_evaluation_metrics(self):
        """Test model evaluation metrics calculation."""
        from backend.app.tasks.train_model import calculate_metrics
        
        # Mock predictions and true labels
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        y_pred_proba = [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], 
                       [0.2, 0.8], [0.9, 0.1], [0.6, 0.4]]
        
        metrics = calculate_metrics(
            y_true, y_pred, y_pred_proba, 
            model_type='classification'
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # All metrics should be between 0 and 1
        for metric_value in metrics.values():
            assert 0 <= metric_value <= 1