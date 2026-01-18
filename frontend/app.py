"""Main Streamlit application."""
import streamlit as st
from components.auth import is_authenticated, get_user_email, logout
from components.api_client import APIClient


# Page configuration
st.set_page_config(
    page_title="LangChain ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

# Sidebar
with st.sidebar:
    st.title("🤖 ML Platform")
    
    if is_authenticated():
        st.success(f"👤 {get_user_email()}")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        
        if st.button("📊 Dashboard", use_container_width=True):
            st.switch_page("pages/9_📈_Dashboard.py")
        
        if st.button("📤 Upload Dataset", use_container_width=True):
            st.switch_page("pages/3_📤_Upload_Dataset.py")
        
        if st.button("🔍 Explore Data", use_container_width=True):
            st.switch_page("pages/4_🔍_Explore_Data.py")
        
        if st.button("🤖 Train Model", use_container_width=True):
            st.switch_page("pages/5_🤖_Train_Model.py")
        
        if st.button("📊 View Results", use_container_width=True):
            st.switch_page("pages/6_📊_Training_Results.py")
        
        if st.button("💬 AI Assistant", use_container_width=True):
            st.switch_page("pages/7_💬_AI_Assistant.py")
        
        if st.button("🎯 Make Predictions", use_container_width=True):
            st.switch_page("pages/8_🎯_Make_Predictions.py")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    else:
        st.info("Please login or register")
        
        if st.button("🔐 Login", use_container_width=True):
            st.switch_page("pages/1_🔐_Login.py")
        
        if st.button("📝 Register", use_container_width=True):
            st.switch_page("pages/2_📝_Register.py")

# Main content
st.markdown('<div class="main-header">🤖 LangChain ML Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Machine Learning Platform with Docker & Streamlit</div>', unsafe_allow_html=True)

if is_authenticated():
    st.success("✅ You are logged in! Use the sidebar to navigate.")
    
    # Features overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Data Management")
        st.markdown("""
        - Upload datasets (CSV, Excel, JSON, Parquet)
        - Data profiling and visualization
        - Dataset exploration
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 🤖 ML Training")
        st.markdown("""
        - Multiple algorithms
        - Async training with Celery
        - Real-time progress tracking
        - Model evaluation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 💬 AI Assistant")
        st.markdown("""
        - LangChain-powered chat
        - Dataset analysis
        - Model recommendations
        - Performance diagnostics
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats
    try:
        datasets = st.session_state.api_client.list_datasets()
        models = st.session_state.api_client.list_models()
        jobs = st.session_state.api_client.list_training_jobs()
        
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Datasets", len(datasets))
        
        with col2:
            st.metric("Trained Models", len(models))
        
        with col3:
            active_jobs = len([j for j in jobs if j['status'] in ['pending', 'running']])
            st.metric("Active Jobs", active_jobs)
        
        with col4:
            completed_jobs = len([j for j in jobs if j['status'] == 'completed'])
            st.metric("Completed Jobs", completed_jobs)
    
    except Exception as e:
        st.info("Connect to the backend to see your statistics")

else:
    st.info("👈 Please login or register using the sidebar to get started!")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Data Management
        - 📤 Upload multiple file formats
        - 🔍 Comprehensive data profiling
        - 📊 Interactive visualizations
        - 🗑️ Dataset management
        
        #### ML Training
        - 🤖 Multiple algorithms (RF, XGBoost, SVM, etc.)
        - ⚡ Asynchronous training with Celery
        - 📈 Real-time progress tracking
        - 🎯 Comprehensive model evaluation
        """)
    
    with col2:
        st.markdown("""
        #### AI-Powered Assistant
        - 💬 LangChain integration
        - 🔍 Intelligent dataset analysis
        - 💡 Model recommendations
        - 🩺 Performance diagnostics
        
        #### Predictions & Deployment
        - 🎯 Single and batch predictions
        - 📊 Prediction history
        - 💾 Model versioning
        - 📥 Export capabilities
        """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Register** an account or **Login** if you already have one
    2. **Upload** your dataset
    3. **Explore** and understand your data
    4. **Train** a machine learning model
    5. **Evaluate** results and make predictions
    6. **Chat** with the AI assistant for help!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🤖 LangChain ML Platform v1.0.0</p>
    <p>Built with FastAPI, Celery, MongoDB, LangChain & Streamlit</p>
</div>
""", unsafe_allow_html=True)

