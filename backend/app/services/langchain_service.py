"""LangChain service for AI-powered ML assistance."""
from typing import Dict, Any, List
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from bson import ObjectId

from app.config import settings
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.training_job import TrainingJob
from app.services.data_service import DataService


class LangChainService:
    """Service for LangChain-powered ML assistance."""
    
    def __init__(self):
        """Initialize LangChain service with LLM and tools."""
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.tools = self._create_tools()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def _create_tools(self) -> List[Tool]:
        """Create custom tools for ML operations."""
        return [
            Tool(
                name="analyze_dataset",
                func=self.analyze_dataset_tool,
                description="Analyzes a dataset by ID. Input should be a dataset ID string. Returns comprehensive statistics, column information, and data insights."
            ),
            Tool(
                name="recommend_model",
                func=self.recommend_model_tool,
                description="Recommends ML algorithms based on problem description and dataset characteristics. Input should be a JSON string with 'problem_description' and 'dataset_id' keys."
            ),
            Tool(
                name="explain_results",
                func=self.explain_results_tool,
                description="Explains model training results in human-readable format. Input should be a model ID string. Returns interpretation of metrics and performance."
            ),
            Tool(
                name="suggest_features",
                func=self.suggest_features_tool,
                description="Suggests feature engineering techniques for a dataset. Input should be a dataset ID string. Returns recommendations for improving model performance."
            ),
            Tool(
                name="diagnose_model",
                func=self.diagnose_model_tool,
                description="Diagnoses model performance issues and suggests improvements. Input should be a model ID string. Returns debugging suggestions and optimization tips."
            )
        ]
    
    async def analyze_dataset_tool(self, dataset_id: str) -> str:
        """Tool to analyze a dataset."""
        try:
            dataset = await Dataset.get(ObjectId(dataset_id))
            if not dataset:
                return f"Dataset with ID {dataset_id} not found."
            
            profile = DataService.profile_dataset(dataset.filepath)
            
            analysis = f"""
Dataset Analysis for '{dataset.filename}':

📊 Basic Information:
- Total Rows: {profile['num_rows']:,}
- Total Columns: {profile['num_columns']}
- File Size: {dataset.size / 1024:.2f} KB

📋 Column Details:
"""
            for col, info in profile['column_info'].items():
                analysis += f"\n  {col}:"
                analysis += f"\n    - Type: {info['dtype']}"
                analysis += f"\n    - Unique Values: {info['unique_values']}"
                analysis += f"\n    - Missing: {info['missing_count']} ({info['missing_percentage']:.1f}%)"
            
            analysis += "\n\n💡 Insights:"
            
            # Check for missing values
            missing_cols = [col for col, count in profile['missing_values'].items() if count > 0]
            if missing_cols:
                analysis += f"\n  - {len(missing_cols)} columns have missing values"
            else:
                analysis += "\n  - No missing values detected"
            
            # Check for high cardinality
            high_card_cols = [col for col, info in profile['column_info'].items() 
                            if info['unique_values'] > profile['num_rows'] * 0.5]
            if high_card_cols:
                analysis += f"\n  - High cardinality columns (may need encoding): {', '.join(high_card_cols)}"
            
            return analysis
        
        except Exception as e:
            return f"Error analyzing dataset: {str(e)}"
    
    async def recommend_model_tool(self, input_str: str) -> str:
        """Tool to recommend ML models."""
        import json
        
        try:
            # Parse input
            input_data = json.loads(input_str)
            problem_desc = input_data.get('problem_description', '')
            dataset_id = input_data.get('dataset_id', '')
            
            dataset = await Dataset.get(ObjectId(dataset_id))
            if not dataset:
                return "Dataset not found."
            
            # Analyze dataset characteristics
            profile = DataService.profile_dataset(dataset.filepath)
            num_rows = profile['num_rows']
            num_features = profile['num_columns']
            
            recommendations = f"""
Model Recommendations:

Problem: {problem_desc}
Dataset Size: {num_rows:,} rows, {num_features} features

🎯 Recommended Models:

1. Random Forest
   - Best for: Medium to large datasets with mixed feature types
   - Pros: Handles non-linear relationships, less prone to overfitting
   - Parameters to tune: n_estimators (100-500), max_depth (5-20)

2. XGBoost
   - Best for: Structured data with complex patterns
   - Pros: High accuracy, fast training, handles missing values
   - Parameters to tune: learning_rate (0.01-0.3), max_depth (3-10), n_estimators (100-1000)

3. Logistic Regression (for classification)
   - Best for: Linear relationships, interpretability needed
   - Pros: Fast, interpretable, works well with small datasets
   - Parameters to tune: C (0.001-100), solver (lbfgs, liblinear)

💡 Recommendation: {"Start with Random Forest for a good baseline" if num_rows > 1000 else "Try Logistic Regression first for quick results"}
"""
            return recommendations
        
        except Exception as e:
            return f"Error recommending models: {str(e)}"
    
    async def explain_results_tool(self, model_id: str) -> str:
        """Tool to explain model results."""
        try:
            model = await MLModel.get(ObjectId(model_id))
            if not model:
                return f"Model with ID {model_id} not found."
            
            explanation = f"""
Model Results Explanation:

🤖 Model: {model.model_type.replace('_', ' ').title()}
📊 Problem Type: {model.problem_type.title()}
🎯 Target: {model.target_column}

📈 Performance Metrics:
"""
            
            if model.problem_type == "classification":
                acc = model.metrics.get('accuracy', 0) * 100
                f1 = model.metrics.get('f1_score', 0) * 100
                
                explanation += f"\n  - Accuracy: {acc:.2f}%"
                explanation += f"\n  - F1 Score: {f1:.2f}%"
                
                if acc >= 90:
                    explanation += "\n\n✅ Excellent performance! The model is highly accurate."
                elif acc >= 80:
                    explanation += "\n\n✅ Good performance. The model performs well on most cases."
                elif acc >= 70:
                    explanation += "\n\n⚠️ Moderate performance. Consider feature engineering or trying different models."
                else:
                    explanation += "\n\n❌ Low performance. The model needs improvement."
            else:
                r2 = model.metrics.get('r2', 0)
                rmse = model.metrics.get('rmse', 0)
                
                explanation += f"\n  - R² Score: {r2:.4f}"
                explanation += f"\n  - RMSE: {rmse:.4f}"
                
                if r2 >= 0.9:
                    explanation += "\n\n✅ Excellent fit! The model explains variance very well."
                elif r2 >= 0.7:
                    explanation += "\n\n✅ Good fit. The model captures most patterns."
                elif r2 >= 0.5:
                    explanation += "\n\n⚠️ Moderate fit. Consider adding more features or trying different models."
                else:
                    explanation += "\n\n❌ Poor fit. The model needs significant improvement."
            
            # Feature importance
            if model.feature_importance:
                explanation += "\n\n🔍 Top Important Features:"
                sorted_features = sorted(model.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    explanation += f"\n  {i}. {feature}: {abs(importance):.4f}"
            
            return explanation
        
        except Exception as e:
            return f"Error explaining results: {str(e)}"
    
    async def suggest_features_tool(self, dataset_id: str) -> str:
        """Tool to suggest feature engineering techniques."""
        try:
            dataset = await Dataset.get(ObjectId(dataset_id))
            if not dataset:
                return f"Dataset with ID {dataset_id} not found."
            
            profile = DataService.profile_dataset(dataset.filepath)
            
            suggestions = f"""
Feature Engineering Suggestions for '{dataset.filename}':

🔧 Recommended Techniques:

"""
            
            # Check for numerical features
            numerical_cols = [col for col, info in profile['column_info'].items() 
                            if 'float' in info['dtype'] or 'int' in info['dtype']]
            
            if len(numerical_cols) >= 2:
                suggestions += f"""
1. Feature Interactions
   - Create polynomial features from numerical columns
   - Try: {numerical_cols[0]} * {numerical_cols[1]}
   
2. Scaling & Normalization
   - Standardize numerical features (mean=0, std=1)
   - Min-Max scaling for bounded ranges
"""
            
            # Check for categorical features
            categorical_cols = [col for col, info in profile['column_info'].items() 
                              if 'object' in info['dtype']]
            
            if categorical_cols:
                suggestions += f"""
3. Categorical Encoding
   - One-hot encoding for low cardinality columns
   - Target encoding for high cardinality columns
   - Columns to encode: {', '.join(categorical_cols[:3])}
"""
            
            # Check for missing values
            missing_cols = [col for col, count in profile['missing_values'].items() if count > 0]
            if missing_cols:
                suggestions += f"""
4. Missing Value Handling
   - Impute with mean/median for numerical features
   - Impute with mode for categorical features
   - Consider creating 'is_missing' indicator features
   - Columns with missing data: {', '.join(missing_cols)}
"""
            
            suggestions += """
5. General Tips
   - Remove highly correlated features (>0.95)
   - Create date-based features if timestamps exist
   - Consider log transformation for skewed distributions
"""
            
            return suggestions
        
        except Exception as e:
            return f"Error suggesting features: {str(e)}"
    
    async def diagnose_model_tool(self, model_id: str) -> str:
        """Tool to diagnose model performance issues."""
        try:
            model = await MLModel.get(ObjectId(model_id))
            if not model:
                return f"Model with ID {model_id} not found."
            
            diagnosis = f"""
Model Diagnosis for {model.model_type.replace('_', ' ').title()}:

🔍 Performance Analysis:
"""
            
            if model.problem_type == "classification":
                acc = model.metrics.get('accuracy', 0)
                precision = model.metrics.get('precision', 0)
                recall = model.metrics.get('recall', 0)
                
                diagnosis += f"\n  - Accuracy: {acc:.4f}"
                diagnosis += f"\n  - Precision: {precision:.4f}"
                diagnosis += f"\n  - Recall: {recall:.4f}"
                
                diagnosis += "\n\n💡 Diagnosis & Recommendations:"
                
                if precision > recall + 0.1:
                    diagnosis += "\n  - High precision, low recall: Model is conservative"
                    diagnosis += "\n  - Suggestion: Adjust classification threshold or use class weights"
                elif recall > precision + 0.1:
                    diagnosis += "\n  - High recall, low precision: Model is aggressive"
                    diagnosis += "\n  - Suggestion: Increase regularization or use more features"
                
                if acc < 0.7:
                    diagnosis += "\n  - Low accuracy detected"
                    diagnosis += "\n  - Try: Add more features, use ensemble methods, tune hyperparameters"
            
            else:  # regression
                r2 = model.metrics.get('r2', 0)
                rmse = model.metrics.get('rmse', 0)
                
                diagnosis += f"\n  - R² Score: {r2:.4f}"
                diagnosis += f"\n  - RMSE: {rmse:.4f}"
                
                diagnosis += "\n\n💡 Diagnosis & Recommendations:"
                
                if r2 < 0.5:
                    diagnosis += "\n  - Low R² score: Model doesn't capture patterns well"
                    diagnosis += "\n  - Try: Feature engineering, polynomial features, different model"
                
                if r2 < 0:
                    diagnosis += "\n  - Negative R²: Model performs worse than baseline"
                    diagnosis += "\n  - Critical: Check data quality and feature selection"
            
            diagnosis += """

🎯 General Improvement Strategies:
  1. Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV
  2. Feature Engineering: Create interaction features
  3. Cross-Validation: Ensure robust performance
  4. Ensemble Methods: Combine multiple models
  5. Data Quality: Check for outliers and data leakage
"""
            
            return diagnosis
        
        except Exception as e:
            return f"Error diagnosing model: {str(e)}"
    
    def create_agent(self) -> AgentExecutor:
        """Create a ReAct agent with tools."""
        template = """You are an AI assistant specializing in machine learning and data science. 
You help users analyze datasets, train models, and understand ML results.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
        
        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    async def chat(self, message: str, user_context: Dict[str, Any] = None) -> str:
        """Process a chat message and return response."""
        try:
            agent = self.create_agent()
            
            # Add user context to message if provided
            if user_context:
                context_str = f"\nUser Context: {user_context}\n\n"
                message = context_str + message
            
            response = await agent.ainvoke({"input": message})
            return response.get("output", "I couldn't process that request.")
        
        except Exception as e:
            return f"Error processing message: {str(e)}"

