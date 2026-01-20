"""
Test cases for LangChain service functionality.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from backend.app.services.langchain_service import LangChainService


@pytest.mark.asyncio
class TestLangChainService:
    """Test LangChain service operations."""
    
    def setup_method(self):
        """Set up test LangChain service."""
        self.langchain_service = LangChainService()
    
    @patch('backend.app.services.langchain_service.ChatOpenAI')
    async def test_initialize_chat_model_openai(self, mock_chat_openai):
        """Test initializing OpenAI chat model."""
        mock_model = Mock()
        mock_chat_openai.return_value = mock_model
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            model = await self.langchain_service.initialize_chat_model('openai')
            
            assert model is not None
            mock_chat_openai.assert_called_once()
    
    @patch('backend.app.services.langchain_service.ChatAnthropic')
    async def test_initialize_chat_model_anthropic(self, mock_chat_anthropic):
        """Test initializing Anthropic chat model."""
        mock_model = Mock()
        mock_chat_anthropic.return_value = mock_model
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            model = await self.langchain_service.initialize_chat_model('anthropic')
            
            assert model is not None
            mock_chat_anthropic.assert_called_once()
    
    async def test_initialize_chat_model_unsupported(self):
        """Test initializing unsupported model provider."""
        with pytest.raises(ValueError, match="Unsupported model provider"):
            await self.langchain_service.initialize_chat_model('unsupported')
    
    @patch('backend.app.services.langchain_service.ChatOpenAI')
    async def test_generate_response_simple_question(self, mock_chat_openai):
        """Test generating response to simple question."""
        # Mock the chat model
        mock_model = AsyncMock()
        mock_response = Mock()
        mock_response.content = "Machine learning is a subset of artificial intelligence."
        mock_model.ainvoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            response = await self.langchain_service.generate_response(
                message="What is machine learning?",
                context={}
            )
            
            assert response is not None
            assert "machine learning" in response.lower()
            mock_model.ainvoke.assert_called_once()
    
    @patch('backend.app.services.langchain_service.ChatOpenAI')
    async def test_generate_response_with_dataset_context(self, mock_chat_openai):
        """Test generating response with dataset context."""
        mock_model = AsyncMock()
        mock_response = Mock()
        mock_response.content = "Your dataset has 3 columns and 100 rows."
        mock_model.ainvoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model
        
        context = {
            'dataset_info': {
                'name': 'Test Dataset',
                'columns': ['feature1', 'feature2', 'target'],
                'row_count': 100,
                'statistics': {'feature1': {'mean': 5.0}}
            }
        }
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            response = await self.langchain_service.generate_response(
                message="Analyze my dataset",
                context=context
            )
            
            assert response is not None
            mock_model.ainvoke.assert_called_once()
    
    @patch('backend.app.services.langchain_service.ChatOpenAI')
    async def test_generate_response_with_model_context(self, mock_chat_openai):
        """Test generating response with model context."""
        mock_model = AsyncMock()
        mock_response = Mock()
        mock_response.content = "Your model has 85% accuracy."
        mock_model.ainvoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model
        
        context = {
            'model_info': {
                'name': 'Test Model',
                'model_type': 'classification',
                'algorithm': 'random_forest',
                'metrics': {'accuracy': 0.85, 'f1_score': 0.82}
            }
        }
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            response = await self.langchain_service.generate_response(
                message="Explain my model performance",
                context=context
            )
            
            assert response is not None
            mock_model.ainvoke.assert_called_once()
    
    async def test_format_dataset_context(self):
        """Test formatting dataset context for prompt."""
        dataset_info = {
            'name': 'Sample Dataset',
            'columns': ['age', 'income', 'purchased'],
            'row_count': 1000,
            'statistics': {
                'age': {'mean': 35.5, 'std': 12.2},
                'income': {'mean': 50000, 'std': 15000}
            }
        }
        
        formatted = self.langchain_service.format_dataset_context(dataset_info)
        
        assert 'Sample Dataset' in formatted
        assert 'age' in formatted
        assert 'income' in formatted
        assert '1000 rows' in formatted
    
    async def test_format_model_context(self):
        """Test formatting model context for prompt."""
        model_info = {
            'name': 'Customer Prediction Model',
            'model_type': 'classification',
            'algorithm': 'random_forest',
            'metrics': {
                'accuracy': 0.87,
                'precision': 0.85,
                'recall': 0.88,
                'f1_score': 0.86
            },
            'feature_importance': [
                {'feature': 'income', 'importance': 0.45},
                {'feature': 'age', 'importance': 0.35}
            ]
        }
        
        formatted = self.langchain_service.format_model_context(model_info)
        
        assert 'Customer Prediction Model' in formatted
        assert 'classification' in formatted
        assert 'random_forest' in formatted
        assert '87%' in formatted or '0.87' in formatted
        assert 'income' in formatted
    
    async def test_get_quick_actions(self):
        """Test getting available quick actions."""
        quick_actions = await self.langchain_service.get_quick_actions()
        
        assert isinstance(quick_actions, list)
        assert len(quick_actions) > 0
        
        # Check that actions have required fields
        for action in quick_actions:
            assert 'name' in action
            assert 'description' in action
            assert 'prompt' in action
    
    async def test_suggest_next_steps_dataset(self):
        """Test suggesting next steps for dataset analysis."""
        context = {
            'dataset_info': {
                'name': 'New Dataset',
                'columns': ['feature1', 'feature2', 'target'],
                'row_count': 500
            }
        }
        
        suggestions = await self.langchain_service.suggest_next_steps(context)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest data exploration or model training
        suggestion_text = ' '.join(suggestions).lower()
        assert any(keyword in suggestion_text for keyword in ['explore', 'analyze', 'train', 'model'])
    
    async def test_suggest_next_steps_model(self):
        """Test suggesting next steps for model analysis."""
        context = {
            'model_info': {
                'name': 'Trained Model',
                'model_type': 'classification',
                'metrics': {'accuracy': 0.75}
            }
        }
        
        suggestions = await self.langchain_service.suggest_next_steps(context)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest model improvements or predictions
        suggestion_text = ' '.join(suggestions).lower()
        assert any(keyword in suggestion_text for keyword in ['improve', 'predict', 'evaluate', 'deploy'])
    
    @patch('backend.app.services.langchain_service.ChatOpenAI')
    async def test_error_handling_api_failure(self, mock_chat_openai):
        """Test error handling when API call fails."""
        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_model
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with pytest.raises(Exception):
                await self.langchain_service.generate_response(
                    message="Test message",
                    context={}
                )
    
    async def test_validate_message_input(self):
        """Test message input validation."""
        # Test empty message
        with pytest.raises(ValueError, match="Message cannot be empty"):
            await self.langchain_service.generate_response("", {})
        
        # Test very long message
        long_message = "x" * 10000
        with pytest.raises(ValueError, match="Message too long"):
            await self.langchain_service.generate_response(long_message, {})