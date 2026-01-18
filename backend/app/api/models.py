"""ML Model API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from bson import ObjectId
import joblib
import pandas as pd
from datetime import datetime
from app.api.schemas import MLModelResponse, PredictionRequest, PredictionResponse
from app.models.user import User
from app.models.ml_model import MLModel
from app.models.prediction import Prediction
from app.utils.auth import get_current_user
from app.services.data_service import DataService


router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("/", response_model=List[MLModelResponse])
async def list_models(current_user: User = Depends(get_current_user)):
    """List all trained models for the current user."""
    models = await MLModel.find(MLModel.user_id == current_user.id).sort(-MLModel.created_at).to_list()
    
    return [
        MLModelResponse(
            id=str(model.id),
            job_id=str(model.job_id),
            model_type=model.model_type,
            problem_type=model.problem_type,
            target_column=model.target_column,
            feature_columns=model.feature_columns,
            metrics=model.metrics,
            feature_importance=model.feature_importance,
            created_at=model.created_at,
            version=model.version
        )
        for model in models
    ]


@router.get("/{model_id}", response_model=MLModelResponse)
async def get_model(model_id: str, current_user: User = Depends(get_current_user)):
    """Get model details by ID."""
    model = await MLModel.get(ObjectId(model_id))
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    if model.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this model"
        )
    
    return MLModelResponse(
        id=str(model.id),
        job_id=str(model.job_id),
        model_type=model.model_type,
        problem_type=model.problem_type,
        target_column=model.target_column,
        feature_columns=model.feature_columns,
        metrics=model.metrics,
        feature_importance=model.feature_importance,
        created_at=model.created_at,
        version=model.version
    )


@router.post("/{model_id}/predict", response_model=PredictionResponse)
async def make_prediction(
    model_id: str,
    request: PredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """Make a prediction using a trained model."""
    model = await MLModel.get(ObjectId(model_id))
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    if model.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to use this model"
        )
    
    try:
        # Load the trained model
        trained_model = joblib.load(model.model_path)
        
        # Prepare input data
        input_df = pd.DataFrame([request.input_data])
        
        # Ensure all required features are present
        missing_features = set(model.feature_columns) - set(input_df.columns)
        if missing_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required features: {missing_features}"
            )
        
        # Select only the required features in the correct order
        input_df = input_df[model.feature_columns]
        
        # Encode categorical features if needed
        input_df = DataService.encode_categorical_features(input_df)
        
        # Make prediction
        prediction = trained_model.predict(input_df)[0]
        
        # Get probabilities for classification
        probabilities = None
        if model.problem_type == "classification" and hasattr(trained_model, 'predict_proba'):
            proba = trained_model.predict_proba(input_df)[0]
            probabilities = {
                str(class_label): float(prob) 
                for class_label, prob in zip(trained_model.classes_, proba)
            }
        
        # Save prediction to database
        prediction_record = Prediction(
            model_id=model.id,
            user_id=current_user.id,
            input_data=request.input_data,
            output={"prediction": str(prediction)},
            probabilities=probabilities,
            timestamp=datetime.utcnow()
        )
        await prediction_record.insert()
        
        return PredictionResponse(
            prediction=str(prediction),
            probabilities=probabilities,
            timestamp=prediction_record.timestamp
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

