"""Dataset API endpoints."""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from typing import List
import os
import uuid
import pandas as pd
from datetime import datetime
from bson import ObjectId
from app.api.schemas import DatasetResponse, DatasetProfile
from app.models.user import User
from app.models.dataset import Dataset
from app.utils.auth import get_current_user
from app.config import settings
from app.services.data_service import DataService


router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a new dataset file."""
    # Validate file type
    allowed_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    file_content = await file.read()
    file_size = len(file_content)
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    # Save file
    os.makedirs(settings.DATASET_STORAGE_PATH, exist_ok=True)
    file_id = str(uuid.uuid4())
    filepath = os.path.join(settings.DATASET_STORAGE_PATH, f"{file_id}{file_ext}")
    
    with open(filepath, 'wb') as f:
        f.write(file_content)
    
    # Extract basic dataset info
    try:
        df = DataService.load_dataset(filepath)
        num_rows, num_columns = df.shape
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        column_names = df.columns.tolist()
        status_val = "ready"
        error_msg = None
    except Exception as e:
        num_rows = None
        num_columns = None
        column_types = None
        column_names = None
        status_val = "error"
        error_msg = str(e)
    
    # Create dataset record
    dataset = Dataset(
        user_id=current_user.id,
        filename=file.filename,
        filepath=filepath,
        size=file_size,
        num_rows=num_rows,
        num_columns=num_columns,
        column_types=column_types,
        column_names=column_names,
        upload_date=datetime.utcnow(),
        status=status_val,
        error_message=error_msg
    )
    
    await dataset.insert()
    
    return DatasetResponse(
        id=str(dataset.id),
        filename=dataset.filename,
        size=dataset.size,
        num_rows=dataset.num_rows,
        num_columns=dataset.num_columns,
        column_types=dataset.column_types,
        column_names=dataset.column_names,
        upload_date=dataset.upload_date,
        status=dataset.status
    )


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(current_user: User = Depends(get_current_user)):
    """List all datasets for the current user."""
    datasets = await Dataset.find(Dataset.user_id == current_user.id).sort(-Dataset.upload_date).to_list()
    
    return [
        DatasetResponse(
            id=str(ds.id),
            filename=ds.filename,
            size=ds.size,
            num_rows=ds.num_rows,
            num_columns=ds.num_columns,
            column_types=ds.column_types,
            column_names=ds.column_names,
            upload_date=ds.upload_date,
            status=ds.status
        )
        for ds in datasets
    ]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, current_user: User = Depends(get_current_user)):
    """Get dataset details by ID."""
    dataset = await Dataset.get(ObjectId(dataset_id))
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this dataset"
        )
    
    return DatasetResponse(
        id=str(dataset.id),
        filename=dataset.filename,
        size=dataset.size,
        num_rows=dataset.num_rows,
        num_columns=dataset.num_columns,
        column_types=dataset.column_types,
        column_names=dataset.column_names,
        upload_date=dataset.upload_date,
        status=dataset.status
    )


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    rows: int = 10,
    current_user: User = Depends(get_current_user)
):
    """Preview the first N rows of a dataset."""
    dataset = await Dataset.get(ObjectId(dataset_id))
    
    if not dataset or dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    try:
        df = DataService.load_dataset(dataset.filepath)
        preview_data = df.head(rows).to_dict(orient='records')
        
        return {
            "dataset_id": str(dataset.id),
            "rows_shown": len(preview_data),
            "total_rows": len(df),
            "columns": df.columns.tolist(),
            "data": preview_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading dataset: {str(e)}"
        )


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
async def profile_dataset(dataset_id: str, current_user: User = Depends(get_current_user)):
    """Get detailed profiling information for a dataset."""
    dataset = await Dataset.get(ObjectId(dataset_id))
    
    if not dataset or dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    try:
        profile = DataService.profile_dataset(dataset.filepath)
        return DatasetProfile(
            dataset_id=str(dataset.id),
            **profile
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error profiling dataset: {str(e)}"
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: str, current_user: User = Depends(get_current_user)):
    """Delete a dataset."""
    dataset = await Dataset.get(ObjectId(dataset_id))
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this dataset"
        )
    
    # Delete file from storage
    if os.path.exists(dataset.filepath):
        os.remove(dataset.filepath)
    
    # Delete from database
    await dataset.delete()
    
    return None

