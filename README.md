# 🤖 LangChain ML Platform

A comprehensive, production-ready machine learning platform built with **FastAPI**, **Celery**, **MongoDB**, **LangChain**, and **Streamlit**, all containerized with **Docker**.

## ✨ Features

### 📊 Data Management
- Upload datasets (CSV, Excel, JSON, Parquet)
- Comprehensive data profiling and statistics
- Interactive data visualization
- Dataset management and deletion

### 🤖 ML Training
- Multiple algorithms: Logistic Regression, Random Forest, XGBoost, SVM, Neural Networks
- Asynchronous training with Celery
- Real-time progress tracking
- Comprehensive model evaluation metrics
- Feature importance analysis
- Hyperparameter configuration

### 💬 AI-Powered Assistant
- LangChain integration with custom tools
- Intelligent dataset analysis
- Model recommendations
- Performance diagnostics
- Feature engineering suggestions
- Natural language interaction

### 🎯 Predictions
- Single and batch predictions
- Probability estimates for classification
- Prediction history tracking
- CSV export functionality

### 📈 Dashboard & Analytics
- Overview metrics and statistics
- Model performance comparison
- Training job monitoring
- Activity timeline

## 🏗️ Architecture

```
langchain-ml-platform/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── models/         # Database models (Beanie ODM)
│   │   ├── services/       # Business logic
│   │   ├── tasks/          # Celery tasks
│   │   └── utils/          # Utilities
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # Streamlit frontend
│   ├── pages/             # Streamlit pages
│   ├── components/        # Reusable components
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml     # Docker Compose configuration
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI or Anthropic API key (for LangChain features)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd langchain-ml-platform
```

2. **Create environment file**
```bash
cp .env.example .env
```

3. **Configure environment variables**
Edit `.env` and set your API keys:
```env
OPENAI_API_KEY=your-openai-api-key
# or
ANTHROPIC_API_KEY=your-anthropic-api-key
```

4. **Start the platform**
```bash
docker-compose up -d
```

5. **Access the application**
- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### First Time Setup

1. Navigate to http://localhost:8501
2. Click "Register" to create an account
3. Login with your credentials
4. Upload your first dataset
5. Start training models!

## 📦 Services

The platform consists of 6 Docker containers:

- **MongoDB**: Database for storing users, datasets, models, and jobs
- **Redis**: Message broker and result backend for Celery
- **Backend**: FastAPI REST API
- **Celery Worker**: Async task processing for model training
- **Celery Beat**: Scheduled tasks
- **Frontend**: Streamlit web interface

## 🛠️ Development

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Run Celery Worker Locally

```bash
cd backend
celery -A app.celery_app.celery_app worker --loglevel=info
```

## 📚 API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

### Key Endpoints

- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get JWT token
- `POST /api/v1/datasets/upload` - Upload dataset
- `GET /api/v1/datasets/` - List datasets
- `POST /api/v1/training/start` - Start training job
- `GET /api/v1/training/{job_id}` - Get training status
- `GET /api/v1/models/` - List trained models
- `POST /api/v1/models/{model_id}/predict` - Make prediction
- `POST /api/v1/chat/message` - Chat with AI assistant

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URL` | MongoDB connection string | `mongodb://mongodb:27017` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `SECRET_KEY` | JWT secret key | (required) |
| `OPENAI_API_KEY` | OpenAI API key | (optional) |
| `ANTHROPIC_API_KEY` | Anthropic API key | (optional) |
| `MAX_FILE_SIZE_MB` | Maximum upload size | `100` |

### Docker Compose Customization

Edit `docker-compose.yml` to customize:
- Port mappings
- Resource limits
- Volume mounts
- Environment variables

## 📊 Supported ML Algorithms

### Classification
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (Neural Network)

### Regression
- Random Forest Regressor
- XGBoost Regressor
- Support Vector Regressor (SVR)
- Multi-Layer Perceptron (Neural Network)

## 🧠 LangChain Tools

The AI Assistant has access to the following custom tools:

1. **analyze_dataset**: Comprehensive dataset analysis
2. **recommend_model**: ML model recommendations
3. **explain_results**: Human-readable result explanations
4. **suggest_features**: Feature engineering suggestions
5. **diagnose_model**: Performance diagnostics

## 🔒 Security

- JWT-based authentication
- Password hashing with bcrypt
- User-specific data isolation
- CORS protection
- Input validation with Pydantic

## 📈 Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f celery-worker
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

## 🛑 Stopping the Platform

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (caution: deletes all data)
docker-compose down -v
```

## 🔄 Updating

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

## 🐛 Troubleshooting

### Backend not starting
- Check MongoDB connection
- Verify environment variables
- Check logs: `docker-compose logs backend`

### Celery tasks not running
- Ensure Redis is running
- Check Celery worker logs: `docker-compose logs celery-worker`
- Verify task routing configuration

### Frontend can't connect to backend
- Verify BACKEND_URL environment variable
- Check if backend is healthy: `http://localhost:8000/api/v1/health`
- Check network connectivity between containers

### Database connection issues
- Ensure MongoDB is running: `docker-compose ps`
- Check MongoDB logs: `docker-compose logs mongodb`
- Verify MONGODB_URL configuration


## 🎉 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Celery](https://docs.celeryproject.org/)
- [MongoDB](https://www.mongodb.com/)
- [Redis](https://redis.io/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Plotly](https://plotly.com/)


