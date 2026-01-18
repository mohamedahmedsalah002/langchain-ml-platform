# 🚀 LangChain ML Platform - Complete Setup Guide

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Configuration](#configuration)
5. [First Use](#first-use)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Required
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Disk Space**: Minimum 5GB free space
- **OS**: Linux, macOS, or Windows with WSL2

### Optional
- **OpenAI API Key**: For LangChain AI assistant features
- **Anthropic API Key**: Alternative to OpenAI

## Quick Start

### 1. Navigate to Project Directory
```bash
cd /Users/mo/PycharmProjects/JupyterProject1/langchain-ml-platform
```

### 2. Create Environment File
```bash
# Copy example environment file
cp .env.example .env

# Edit with your favorite editor
nano .env  # or vim, code, etc.
```

### 3. Add Your API Keys (Optional but Recommended)
Edit `.env` and add:
```env
OPENAI_API_KEY=sk-your-openai-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# For production, also change:
SECRET_KEY=your-very-secure-random-string-here
```

### 4. Start the Platform
```bash
./start.sh
```

Or manually:
```bash
docker-compose up -d
```

### 5. Access the Application
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Detailed Setup

### Step 1: Verify Docker Installation
```bash
docker --version
docker-compose --version
```

Expected output:
```
Docker version 20.10.x or higher
Docker Compose version 2.x.x or higher
```

### Step 2: Check Port Availability
Ensure these ports are available:
- `8501` - Frontend (Streamlit)
- `8000` - Backend (FastAPI)
- `27017` - MongoDB
- `6379` - Redis

Check ports:
```bash
# macOS/Linux
lsof -i :8501
lsof -i :8000
lsof -i :27017
lsof -i :6379

# Windows
netstat -an | findstr "8501"
netstat -an | findstr "8000"
```

### Step 3: Configure Environment Variables

#### Minimum Configuration
```env
# .env file
SECRET_KEY=change-this-to-a-random-string
OPENAI_API_KEY=your-key-here
```

#### Full Configuration
```env
# Database
MONGODB_URL=mongodb://mongodb:27017
MONGODB_DB_NAME=ml_platform

# Redis
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

# Authentication
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# LLM API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Application
MAX_FILE_SIZE_MB=100
ALLOWED_ORIGINS=http://localhost:8501,http://frontend:8501

# Paths
DATASET_STORAGE_PATH=/app/data/datasets
MODEL_STORAGE_PATH=/app/data/models
```

### Step 4: Build and Start Services

#### Option A: Using Start Script
```bash
./start.sh
```

#### Option B: Manual Docker Compose
```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Step 5: Verify All Services are Running
```bash
docker-compose ps
```

Expected output - all services should show "Up":
```
NAME                        STATUS
ml-platform-backend         Up
ml-platform-celery-worker   Up
ml-platform-celery-beat     Up
ml-platform-frontend        Up
ml-platform-mongodb         Up
ml-platform-redis           Up
```

### Step 6: Health Check
```bash
# Check backend health
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status":"healthy","timestamp":"...","service":"ml-platform-api"}
```

## Configuration

### Customizing Docker Compose

Edit `docker-compose.yml` to customize:

#### Change Ports
```yaml
frontend:
  ports:
    - "8080:8501"  # Change 8080 to your preferred port

backend:
  ports:
    - "9000:8000"  # Change 9000 to your preferred port
```

#### Resource Limits
```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '1'
        memory: 1G
```

#### Celery Worker Concurrency
```yaml
celery-worker:
  command: celery -A app.celery_app.celery_app worker --loglevel=info --concurrency=8
```

### Production Configuration

For production deployment:

1. **Use strong SECRET_KEY**:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

2. **Enable HTTPS** (use nginx reverse proxy)
3. **Set up database backups**
4. **Configure log rotation**
5. **Use production-grade MongoDB** (with authentication)
6. **Set up monitoring** (Prometheus, Grafana)

## First Use

### 1. Register an Account
1. Navigate to http://localhost:8501
2. Click "Register" in the sidebar
3. Enter email and password
4. Click "Register"

### 2. Login
1. Enter your credentials
2. Click "Login"
3. You'll be redirected to the dashboard

### 3. Upload Your First Dataset
1. Click "Upload Dataset" in the sidebar
2. Choose a CSV, Excel, JSON, or Parquet file
3. Preview the data
4. Click "Upload Dataset"

### 4. Explore Your Data
1. Navigate to "Explore Data"
2. Select your dataset
3. View statistics and visualizations
4. Analyze data quality

### 5. Train Your First Model
1. Navigate to "Train Model"
2. Select your dataset
3. Choose problem type (Classification/Regression)
4. Select target and feature columns
5. Choose a model algorithm
6. Configure hyperparameters
7. Click "Start Training"

### 6. View Results
1. Navigate to "Training Results"
2. Select your training job
3. View metrics and performance
4. Analyze feature importance

### 7. Make Predictions
1. Navigate to "Make Predictions"
2. Select your trained model
3. Enter input values or upload a file
4. Click "Make Prediction"
5. View results and download if needed

### 8. Chat with AI Assistant
1. Navigate to "AI Assistant"
2. Ask questions about your data or models
3. Get recommendations and insights
4. Use quick action buttons for common tasks

## Troubleshooting

### Issue: Containers won't start
```bash
# Check logs for specific service
docker-compose logs backend
docker-compose logs celery-worker
docker-compose logs mongodb

# Common solution: Remove old containers and volumes
docker-compose down -v
docker-compose up -d
```

### Issue: Backend API not accessible
```bash
# Check if backend is running
docker-compose ps backend

# Check backend logs
docker-compose logs backend

# Restart backend
docker-compose restart backend
```

### Issue: Training jobs stuck in "pending"
```bash
# Check Celery worker status
docker-compose logs celery-worker

# Restart Celery worker
docker-compose restart celery-worker

# Check Redis connection
docker-compose exec redis redis-cli ping
```

### Issue: MongoDB connection failed
```bash
# Check MongoDB status
docker-compose ps mongodb

# View MongoDB logs
docker-compose logs mongodb

# Restart MongoDB
docker-compose restart mongodb

# Connect to MongoDB shell
docker-compose exec mongodb mongosh ml_platform
```

### Issue: Frontend can't connect to backend
1. Check BACKEND_URL in frontend environment
2. Verify backend is running: http://localhost:8000/api/v1/health
3. Check Docker network: `docker network inspect ml-platform-network`

### Issue: Out of disk space
```bash
# Check Docker disk usage
docker system df

# Clean up unused resources
docker system prune -a

# Remove old volumes (caution: deletes data)
docker volume prune
```

### Issue: Port already in use
```bash
# Find process using port
lsof -i :8501  # or :8000, :27017, :6379

# Kill the process or change port in docker-compose.yml
```

## Maintenance

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f celery-worker

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Restart Services
```bash
# All services
docker-compose restart

# Specific service
docker-compose restart backend
docker-compose restart celery-worker
```

### Update Platform
```bash
# Pull latest changes (if using git)
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Backup Data
```bash
# Backup MongoDB
docker-compose exec mongodb mongodump --db ml_platform --out /backup

# Copy backup from container
docker cp ml-platform-mongodb:/backup ./mongodb_backup

# Backup uploaded datasets and models
tar -czf data_backup.tar.gz data/
```

### Stop Platform
```bash
./stop.sh

# Or manually
docker-compose down

# To also remove volumes (caution: deletes all data)
docker-compose down -v
```

## Performance Optimization

### 1. Increase Celery Workers
```yaml
celery-worker:
  command: celery -A app.celery_app.celery_app worker --loglevel=info --concurrency=8
  deploy:
    replicas: 2  # Run 2 worker instances
```

### 2. Add MongoDB Indexes
Already included in the models, but you can add more:
```python
# In your model file
class Settings:
    indexes = [
        "user_id",
        "created_at",
        [("user_id", 1), ("created_at", -1)]  # Compound index
    ]
```

### 3. Enable Redis Persistence
```yaml
redis:
  command: redis-server --appendonly yes
```

### 4. Use Production ASGI Server
The platform already uses uvicorn, but for production:
```yaml
backend:
  command: gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Security Checklist

- [ ] Change SECRET_KEY to a strong random value
- [ ] Enable MongoDB authentication
- [ ] Use HTTPS (reverse proxy with Let's Encrypt)
- [ ] Set up firewall rules
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] API rate limiting
- [ ] Input validation (already implemented)
- [ ] CORS configuration (already implemented)

## Support

### Common Commands Cheat Sheet
```bash
# Start platform
./start.sh

# Stop platform
./stop.sh

# View logs
docker-compose logs -f

# Restart service
docker-compose restart backend

# Check status
docker-compose ps

# Health check
curl http://localhost:8000/api/v1/health

# Access MongoDB
docker-compose exec mongodb mongosh ml_platform

# Access Redis
docker-compose exec redis redis-cli

# Clean up
docker-compose down
docker system prune
```

### Getting Help
- Check logs: `docker-compose logs -f`
- API documentation: http://localhost:8000/docs
- GitHub Issues: Create an issue with logs and error messages

---

**You're all set! Happy Machine Learning! 🚀🤖**

