# ⚡ Quick Start Guide

## 🚀 Start in 60 Seconds

```bash
# 1. Navigate to project
cd /Users/mo/PycharmProjects/JupyterProject1/langchain-ml-platform

# 2. Configure (optional - works without API keys)
cp .env.example .env

# 3. Start everything
./start.sh

# 4. Open browser
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

## 🎯 First Use

1. **Register**: Click "Register" → Enter email/password
2. **Login**: Enter credentials
3. **Upload**: Choose a CSV/Excel file
4. **Explore**: View data statistics and charts
5. **Train**: Select model → Configure → Start training
6. **Results**: View metrics and performance
7. **Predict**: Use model for predictions
8. **Chat**: Ask AI assistant for help

## 📋 Essential Commands

```bash
# Start platform
./start.sh

# Stop platform
./stop.sh

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Restart a service
docker-compose restart backend

# Clean everything (removes data!)
docker-compose down -v
```

## 🔧 Configuration (Optional)

### Add AI Assistant Features
Edit `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
```

### Change Ports
Edit `docker-compose.yml`:
```yaml
frontend:
  ports:
    - "8080:8501"  # Change 8080 to your port
```

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:8501 | Streamlit UI |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Health Check | http://localhost:8000/api/v1/health | Status |

## 🎓 Example Workflow

### Train Your First Model

1. **Prepare Data**
   - CSV with headers
   - Numerical or categorical features
   - Target column for prediction

2. **Upload**
   ```
   Navigate to: Upload Dataset
   Choose file → Preview → Upload
   ```

3. **Configure Training**
   ```
   Problem Type: Classification or Regression
   Target: Column to predict
   Features: Input columns
   Model: Random Forest (recommended for beginners)
   ```

4. **Train**
   ```
   Click "Start Training"
   Watch real-time progress
   Wait for completion (~30 seconds to 5 minutes)
   ```

5. **Evaluate**
   ```
   View metrics (accuracy, F1, etc.)
   Check feature importance
   Analyze confusion matrix
   ```

6. **Predict**
   ```
   Enter values for features
   Click "Make Prediction"
   View results and confidence
   ```

## 💡 Tips

- **Dataset Format**: CSV works best for beginners
- **Features**: Start with 3-10 features
- **Model**: Random Forest is reliable for most cases
- **Training Time**: 100-1000 rows: <1 min, 10K+ rows: 2-5 min
- **AI Assistant**: Ask "Analyze my dataset" after upload

## 🚨 Troubleshooting

### Platform won't start
```bash
docker-compose down -v
docker-compose up -d
```

### Can't access frontend
```bash
# Check if running
docker-compose ps

# Restart frontend
docker-compose restart frontend
```

### Training stuck
```bash
# Restart Celery worker
docker-compose restart celery-worker
```

## 📞 Need Help?

1. Check logs: `docker-compose logs -f backend`
2. Verify health: `curl http://localhost:8000/api/v1/health`
3. Read full documentation: `README.md` and `SETUP_GUIDE.md`

## 🎉 That's It!

You're ready to:
- ✅ Upload datasets
- ✅ Train ML models
- ✅ Make predictions
- ✅ Chat with AI
- ✅ Analyze data

**Happy Machine Learning! 🚀**

