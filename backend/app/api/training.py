"""Training job API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from bson import ObjectId
from app.api.schemas import TrainingJobCreate, TrainingJobResponse
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.utils.auth import get_current_user
from app.tasks.train_model import train_model_task
from app.tasks.train_model_fixed import train_text_classification_task


router = APIRouter(prefix="/api/v1/training", tags=["Training"])


# List training jobs - MUST come before /{job_id} route
@router.get("/", response_model=List[TrainingJobResponse])
async def list_training_jobs(current_user: User = Depends(get_current_user)):
    """List all training jobs for the current user."""
    jobs = await TrainingJob.find(TrainingJob.user_id == current_user.id).sort(-TrainingJob.started_at).to_list()
    
    return [
        TrainingJobResponse(
            id=str(job.id),
            dataset_id=str(job.dataset_id),
            model_type=job.model_type,
            problem_type=job.problem_type,
            target_column=job.target_column,
            feature_columns=job.feature_columns,
            status=job.status,
            progress=job.progress,
            current_stage=job.current_stage,
            parameters=job.parameters,
            metrics=job.metrics,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message
        )
        for job in jobs
    ]


@router.post("/start", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def start_training(
    job_data: TrainingJobCreate,
    current_user: User = Depends(get_current_user)
):
    """Start a new training job."""
    # Verify dataset exists and belongs to user
    dataset = await Dataset.get(ObjectId(job_data.dataset_id))
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to use this dataset"
        )
    
    if dataset.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset is not ready for training. Status: {dataset.status}"
        )
    
    # Create training job record
    job = TrainingJob(
        dataset_id=dataset.id,
        user_id=current_user.id,
        model_type=job_data.model_type,
        problem_type=job_data.problem_type,
        target_column=job_data.target_column,
        feature_columns=job_data.feature_columns,
        parameters=job_data.parameters,
        status="pending"
    )
    
    await job.insert()
    
    # Start async training task
    # Use fixed text classification task for text data (like payload classification)
    is_text_classification = (
        job_data.problem_type == "classification" and 
        len(job_data.feature_columns) == 1 and
        any(col.lower() in ['payload', 'text', 'message', 'content', 'description'] 
            for col in job_data.feature_columns)
    )
    
    if is_text_classification:
        # Use the fixed text classification task
        task = train_text_classification_task.delay(
            job_id=str(job.id),
            dataset_path=dataset.filepath,
            target_column=job_data.target_column,
            text_column=job_data.feature_columns[0],  # Single text column
            model_name=getattr(job_data, 'model_name', f"Text Classification Model"),
            parameters=job_data.parameters
        )
    else:
        # Use the standard training task for numerical/categorical data
        task = train_model_task.delay(
            job_id=str(job.id),
            dataset_path=dataset.filepath,
            target_column=job_data.target_column,
            feature_columns=job_data.feature_columns,
            model_type=job_data.model_type,
            problem_type=job_data.problem_type,
            parameters=job_data.parameters,
            test_size=job_data.test_size
        )
    
    # Update job with Celery task ID
    job.celery_task_id = task.id
    await job.save()
    
    return TrainingJobResponse(
        id=str(job.id),
        dataset_id=str(job.dataset_id),
        model_type=job.model_type,
        problem_type=job.problem_type,
        target_column=job.target_column,
        feature_columns=job.feature_columns,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        parameters=job.parameters,
        metrics=job.metrics,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message
    )


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str, current_user: User = Depends(get_current_user)):
    """Get training job status and details."""
    job = await TrainingJob.get(ObjectId(job_id))
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )
    
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this training job"
        )
    
    return TrainingJobResponse(
        id=str(job.id),
        dataset_id=str(job.dataset_id),
        model_type=job.model_type,
        problem_type=job.problem_type,
        target_column=job.target_column,
        feature_columns=job.feature_columns,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        parameters=job.parameters,
        metrics=job.metrics,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message
    )
