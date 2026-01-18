"""Model training page."""
import streamlit as st
from components.api_client import APIClient
from components.auth import require_auth
import time

st.set_page_config(page_title="Train Model", page_icon="🤖", layout="wide")

require_auth()

if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

st.title("🤖 Train Machine Learning Model")
st.markdown("Configure and start training a new model")
st.markdown("---")

# Fetch datasets
try:
    datasets = st.session_state.api_client.list_datasets()
    
    if not datasets:
        st.info("📭 No datasets found. Please upload a dataset first.")
        if st.button("Upload Dataset"):
            st.switch_page("pages/3_📤_Upload_Dataset.py")
        st.stop()
    
    # Step 1: Dataset Selection
    st.subheader("1️⃣ Select Dataset")
    
    dataset_options = {f"{ds['filename']}": ds['id'] for ds in datasets if ds['status'] == 'ready'}
    
    if not dataset_options:
        st.warning("No ready datasets available. Please upload a dataset.")
        st.stop()
    
    # Pre-select if coming from explore page
    default_idx = 0
    if 'selected_dataset_id' in st.session_state:
        for idx, (name, ds_id) in enumerate(dataset_options.items()):
            if ds_id == st.session_state.selected_dataset_id:
                default_idx = idx
                break
    
    selected_name = st.selectbox("Choose dataset", list(dataset_options.keys()), index=default_idx)
    selected_dataset_id = dataset_options[selected_name]
    
    # Get dataset details
    selected_dataset = next(ds for ds in datasets if ds['id'] == selected_dataset_id)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{selected_dataset.get('num_rows', 0):,}")
    with col2:
        st.metric("Columns", selected_dataset.get('num_columns', 0))
    with col3:
        st.metric("Status", selected_dataset.get('status', 'unknown').capitalize())
    
    st.markdown("---")
    
    # Step 2: Problem Configuration
    st.subheader("2️⃣ Problem Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        problem_type = st.selectbox(
            "Problem Type",
            ["classification", "regression"],
            help="Classification for categorical targets, Regression for numerical targets"
        )
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05, help="Proportion of data to use for testing")
    
    # Get column names
    columns = selected_dataset.get('column_names', [])
    
    if not columns:
        st.warning("Could not retrieve column names from dataset")
        st.stop()
    
    target_column = st.selectbox("Target Column", columns, help="The variable you want to predict")
    
    feature_columns = st.multiselect(
        "Feature Columns",
        [col for col in columns if col != target_column],
        default=[col for col in columns if col != target_column],
        help="Input variables for prediction"
    )
    
    if not feature_columns:
        st.warning("⚠️ Please select at least one feature column")
        st.stop()
    
    st.markdown("---")
    
    # Step 3: Model Selection
    st.subheader("3️⃣ Model Selection")
    
    if problem_type == "classification":
        model_options = {
            "Logistic Regression": "logistic_regression",
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
            "Support Vector Machine": "svm",
            "Neural Network": "neural_network"
        }
    else:
        model_options = {
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
            "Support Vector Machine": "svm",
            "Neural Network": "neural_network"
        }
    
    selected_model_name = st.selectbox("Choose Model", list(model_options.keys()))
    selected_model_type = model_options[selected_model_name]
    
    # Model descriptions
    model_descriptions = {
        "logistic_regression": "Fast and interpretable, works well for linearly separable data",
        "random_forest": "Ensemble method, handles non-linear relationships well, less prone to overfitting",
        "xgboost": "Gradient boosting, high performance, good for complex patterns",
        "svm": "Effective in high-dimensional spaces, works well for classification",
        "neural_network": "Deep learning, can capture complex non-linear patterns"
    }
    
    st.info(f"ℹ️ {model_descriptions.get(selected_model_type, '')}")
    
    st.markdown("---")
    
    # Step 4: Hyperparameters
    st.subheader("4️⃣ Hyperparameters")
    
    parameters = {}
    
    if selected_model_type == "logistic_regression":
        col1, col2 = st.columns(2)
        with col1:
            parameters['C'] = st.number_input("Regularization (C)", 0.001, 100.0, 1.0, 0.1)
        with col2:
            parameters['max_iter'] = st.number_input("Max Iterations", 100, 10000, 1000, 100)
    
    elif selected_model_type == "random_forest":
        col1, col2, col3 = st.columns(3)
        with col1:
            parameters['n_estimators'] = st.number_input("Number of Trees", 10, 1000, 100, 10)
        with col2:
            parameters['max_depth'] = st.number_input("Max Depth", 1, 50, 10, 1)
        with col3:
            parameters['min_samples_split'] = st.number_input("Min Samples Split", 2, 20, 2, 1)
    
    elif selected_model_type == "xgboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            parameters['n_estimators'] = st.number_input("Number of Estimators", 10, 1000, 100, 10)
        with col2:
            parameters['max_depth'] = st.number_input("Max Depth", 1, 20, 6, 1)
        with col3:
            parameters['learning_rate'] = st.number_input("Learning Rate", 0.01, 1.0, 0.1, 0.01)
    
    elif selected_model_type == "svm":
        col1, col2 = st.columns(2)
        with col1:
            parameters['C'] = st.number_input("Regularization (C)", 0.001, 100.0, 1.0, 0.1)
        with col2:
            parameters['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
    
    elif selected_model_type == "neural_network":
        col1, col2, col3 = st.columns(3)
        with col1:
            hidden_layer_size = st.number_input("Hidden Layer Size", 10, 500, 100, 10)
            parameters['hidden_layer_sizes'] = (hidden_layer_size,)
        with col2:
            parameters['max_iter'] = st.number_input("Max Iterations", 100, 10000, 1000, 100)
        with col3:
            parameters['alpha'] = st.number_input("Alpha (Regularization)", 0.0001, 1.0, 0.0001, 0.0001, format="%.4f")
    
    st.markdown("---")
    
    # Training button
    st.subheader("5️⃣ Start Training")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Training", use_container_width=True, type="primary"):
            try:
                # Prepare training job data
                job_data = {
                    "dataset_id": selected_dataset_id,
                    "model_type": selected_model_type,
                    "problem_type": problem_type,
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                    "parameters": parameters,
                    "test_size": test_size
                }
                
                with st.spinner("Starting training job..."):
                    response = st.session_state.api_client.start_training(job_data)
                    
                    st.success("✅ Training job started successfully!")
                    st.balloons()
                    
                    job_id = response['id']
                    
                    # Show job details
                    st.json(response)
                    
                    st.info("🔄 Training is running asynchronously. You can monitor progress in the Training Results page.")
                    
                    # Real-time progress tracking
                    st.markdown("### 📊 Training Progress")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Poll for updates
                    for _ in range(60):  # Poll for up to 60 seconds
                        time.sleep(1)
                        
                        try:
                            job_status = st.session_state.api_client.get_training_job(job_id)
                            
                            progress = job_status.get('progress', 0)
                            status = job_status.get('status', 'unknown')
                            stage = job_status.get('current_stage', '')
                            
                            progress_bar.progress(progress / 100)
                            status_text.text(f"Status: {status.upper()} | Stage: {stage} | Progress: {progress}%")
                            
                            if status in ['completed', 'failed']:
                                break
                        
                        except:
                            break
                    
                    # Navigate to results
                    if st.button("View Training Results"):
                        st.switch_page("pages/6_📊_Training_Results.py")
            
            except Exception as e:
                st.error(f"❌ Error starting training: {str(e)}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")

