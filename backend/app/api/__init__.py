"""API package - REST API endpoints for the ML Platform.

This package contains all API route handlers organized by resource:
- auth: Authentication endpoints (login, register)
- datasets: Dataset management endpoints
- training: Training job endpoints
- models: Model and prediction endpoints
- chat: LangChain chat interface endpoints
"""

__all__ = ['auth', 'datasets', 'training', 'models', 'chat', 'schemas']

