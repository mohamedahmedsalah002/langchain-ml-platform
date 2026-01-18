#!/bin/bash

echo "🚀 Starting LangChain ML Platform..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✅ Please edit .env file and add your API keys before continuing."
    echo "   Press Ctrl+C to exit and edit, or Enter to continue with defaults."
    read
fi

# Start Docker Compose
echo "🐳 Starting Docker containers..."
docker compose up -d

echo ""
echo "✅ Platform started successfully!"
echo ""
echo "📍 Access points:"
echo "   - Frontend (Streamlit): http://localhost:8501"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "📊 View logs:"
echo "   docker compose logs -f"
echo ""
echo "🛑 Stop platform:"
echo "   docker compose down"
echo ""

