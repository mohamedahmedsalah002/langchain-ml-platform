"""API client for backend communication."""
import requests
import os
from typing import Dict, Any, Optional, List
import streamlit as st


class APIClient:
    """Client for interacting with the backend API."""
    
    def __init__(self, base_url: str = None):
        """Initialize API client."""
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        self.session = requests.Session()
    
    def _get_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": "application/json"}
        
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif "token" in st.session_state:
            headers["Authorization"] = f"Bearer {st.session_state.token}"
        
        return headers
    
    def register(self, email: str, password: str) -> Dict[str, Any]:
        """Register a new user."""
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/register",
            json={"email": email, "password": password}
        )
        response.raise_for_status()
        return response.json()
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Login and get access token."""
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"email": email, "password": password}
        )
        response.raise_for_status()
        return response.json()
    
    def upload_dataset(self, file, token: str = None) -> Dict[str, Any]:
        """Upload a dataset file."""
        files = {"file": file}
        headers = {"Authorization": f"Bearer {token or st.session_state.token}"}
        
        response = self.session.post(
            f"{self.base_url}/api/v1/datasets/upload",
            files=files,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    def list_datasets(self, token: str = None) -> List[Dict[str, Any]]:
        """List all datasets."""
        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def get_dataset(self, dataset_id: str, token: str = None) -> Dict[str, Any]:
        """Get dataset details."""
        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_id}",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def preview_dataset(self, dataset_id: str, rows: int = 10, token: str = None) -> Dict[str, Any]:
        """Preview dataset."""
        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_id}/preview?rows={rows}",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def profile_dataset(self, dataset_id: str, token: str = None) -> Dict[str, Any]:
        """Get dataset profile."""
        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_id}/profile",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def delete_dataset(self, dataset_id: str, token: str = None):
        """Delete a dataset."""
        response = self.session.delete(
            f"{self.base_url}/api/v1/datasets/{dataset_id}",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
    
    def start_training(self, job_data: Dict[str, Any], token: str = None) -> Dict[str, Any]:
        """Start a training job."""
        response = self.session.post(
            f"{self.base_url}/api/v1/training/start",
            json=job_data,
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def get_training_job(self, job_id: str, token: str = None) -> Dict[str, Any]:
        """Get training job status."""
        response = self.session.get(
            f"{self.base_url}/api/v1/training/{job_id}",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def list_training_jobs(self, token: str = None) -> List[Dict[str, Any]]:
        """List all training jobs."""
        response = self.session.get(
            f"{self.base_url}/api/v1/training/",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def list_models(self, token: str = None) -> List[Dict[str, Any]]:
        """List all trained models."""
        response = self.session.get(
            f"{self.base_url}/api/v1/models/",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def get_model(self, model_id: str, token: str = None) -> Dict[str, Any]:
        """Get model details."""
        response = self.session.get(
            f"{self.base_url}/api/v1/models/{model_id}",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def predict(self, model_id: str, input_data: Dict[str, Any], token: str = None) -> Dict[str, Any]:
        """Make a prediction."""
        response = self.session.post(
            f"{self.base_url}/api/v1/models/{model_id}/predict",
            json={"input_data": input_data},
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def send_chat_message(self, message: str, session_id: str = None, token: str = None) -> Dict[str, Any]:
        """Send a chat message."""
        data = {"content": message}
        if session_id:
            data["session_id"] = session_id
        
        response = self.session.post(
            f"{self.base_url}/api/v1/chat/message",
            json=data,
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def get_chat_history(self, session_id: str, token: str = None) -> Dict[str, Any]:
        """Get chat history."""
        response = self.session.get(
            f"{self.base_url}/api/v1/chat/history/{session_id}",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def list_chat_sessions(self, token: str = None) -> List[Dict[str, Any]]:
        """List chat sessions."""
        response = self.session.get(
            f"{self.base_url}/api/v1/chat/sessions",
            headers=self._get_headers(token)
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()

