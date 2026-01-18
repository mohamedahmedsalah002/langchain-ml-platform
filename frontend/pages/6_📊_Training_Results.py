"""Training results page."""
import streamlit as st
from components.api_client import APIClient
from components.auth import require_auth
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Training Results", page_icon="📊", layout="wide")

require_auth()

if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

st.title("📊 Training Results")
st.markdown("View and analyze your training jobs and model performance")
st.markdown("---")

# Fetch training jobs
try:
    jobs = st.session_state.api_client.list_training_jobs()
    
    if not jobs:
        st.info("📭 No training jobs found. Start training a model first.")
        if st.button("Train a Model"):
            st.switch_page("pages/5_🤖_Train_Model.py")
        st.stop()
    
    # Filter options
    col1, col2 = st.columns([1, 3])
    
    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            ["pending", "running", "completed", "failed"],
            default=["completed", "running"]
        )
    
    # Filter jobs
    filtered_jobs = [j for j in jobs if j['status'] in status_filter] if status_filter else jobs
    
    if not filtered_jobs:
        st.warning("No jobs match the selected filters")
        st.stop()
    
    st.markdown(f"**Showing {len(filtered_jobs)} of {len(jobs)} jobs**")
    
    st.markdown("---")
    
    # Summary metrics
    st.subheader("📈 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_jobs = len(jobs)
        st.metric("Total Jobs", total_jobs)
    
    with col2:
        completed = len([j for j in jobs if j['status'] == 'completed'])
        st.metric("Completed", completed)
    
    with col3:
        running = len([j for j in jobs if j['status'] == 'running'])
        st.metric("Running", running)
    
    with col4:
        failed = len([j for j in jobs if j['status'] == 'failed'])
        st.metric("Failed", failed)
    
    st.markdown("---")
    
    # Job list
    st.subheader("🔍 Training Jobs")
    
    # Create jobs table
    jobs_data = []
    for job in filtered_jobs:
        status_emoji = {
            'completed': '✅',
            'running': '🔄',
            'pending': '⏳',
            'failed': '❌'
        }
        
        jobs_data.append({
            'ID': job['id'][:8] + '...',
            'Model': job['model_type'].replace('_', ' ').title(),
            'Status': f"{status_emoji.get(job['status'], '❓')} {job['status'].capitalize()}",
            'Progress': f"{job['progress']}%",
            'Target': job['target_column'],
            'Started': job.get('started_at', 'N/A')
        })
    
    jobs_df = pd.DataFrame(jobs_data)
    st.dataframe(jobs_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Job details
    st.subheader("📋 Job Details")
    
    # Select job to view details
    job_options = {
        f"{j['model_type'].replace('_', ' ').title()} - {j['id'][:8]}... ({j['status']})": j['id'] 
        for j in filtered_jobs
    }
    
    selected_job_name = st.selectbox("Select a job to view details", list(job_options.keys()))
    selected_job_id = job_options[selected_job_name]
    
    # Get job details
    try:
        job = st.session_state.api_client.get_training_job(selected_job_id)
        
        # Job information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", job['model_type'].replace('_', ' ').title())
        with col2:
            st.metric("Problem Type", job['problem_type'].capitalize())
        with col3:
            st.metric("Status", job['status'].capitalize())
        with col4:
            st.metric("Progress", f"{job['progress']}%")
        
        # Show progress bar
        st.progress(job['progress'] / 100)
        
        if job.get('current_stage'):
            st.info(f"📍 Current Stage: {job['current_stage']}")
        
        # Tabs for different information
        tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Metrics", "🔍 Details"])
        
        with tab1:
            st.markdown("#### Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Target Column:**")
                st.code(job['target_column'])
                
                st.markdown("**Feature Columns:**")
                for feature in job['feature_columns']:
                    st.write(f"- {feature}")
            
            with col2:
                st.markdown("**Parameters:**")
                if job.get('parameters'):
                    st.json(job['parameters'])
                else:
                    st.write("Using default parameters")
        
        with tab2:
            st.markdown("#### Performance Metrics")
            
            if job['status'] == 'completed' and job.get('metrics'):
                metrics = job['metrics']
                
                if job['problem_type'] == 'classification':
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        accuracy = metrics.get('accuracy', 0) * 100
                        st.metric("Accuracy", f"{accuracy:.2f}%")
                    
                    with col2:
                        precision = metrics.get('precision', 0) * 100
                        st.metric("Precision", f"{precision:.2f}%")
                    
                    with col3:
                        recall = metrics.get('recall', 0) * 100
                        st.metric("Recall", f"{recall:.2f}%")
                    
                    with col4:
                        f1 = metrics.get('f1_score', 0) * 100
                        st.metric("F1 Score", f"{f1:.2f}%")
                    
                    # Metrics visualization
                    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    metric_values = [
                        metrics.get('accuracy', 0),
                        metrics.get('precision', 0),
                        metrics.get('recall', 0),
                        metrics.get('f1_score', 0)
                    ]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=metric_names, y=metric_values, marker_color='lightblue')
                    ])
                    fig.update_layout(
                        title="Classification Metrics",
                        yaxis_title="Score",
                        yaxis_range=[0, 1]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # regression
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        r2 = metrics.get('r2', 0)
                        st.metric("R² Score", f"{r2:.4f}")
                    
                    with col2:
                        mse = metrics.get('mse', 0)
                        st.metric("MSE", f"{mse:.4f}")
                    
                    with col3:
                        rmse = metrics.get('rmse', 0)
                        st.metric("RMSE", f"{rmse:.4f}")
                    
                    with col4:
                        mae = metrics.get('mae', 0)
                        st.metric("MAE", f"{mae:.4f}")
                
                # Get model details if available
                try:
                    models = st.session_state.api_client.list_models()
                    model = next((m for m in models if m['job_id'] == selected_job_id), None)
                    
                    if model and model.get('feature_importance'):
                        st.markdown("#### Feature Importance")
                        
                        # Sort by importance
                        importance_items = sorted(
                            model['feature_importance'].items(),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:10]
                        
                        features = [item[0] for item in importance_items]
                        importances = [abs(item[1]) for item in importance_items]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=importances, y=features, orientation='h', marker_color='lightgreen')
                        ])
                        fig.update_layout(
                            title="Top 10 Important Features",
                            xaxis_title="Importance",
                            yaxis_title="Feature",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except:
                    pass
            
            elif job['status'] == 'running':
                st.info("⏳ Training in progress... Metrics will be available when completed.")
            
            elif job['status'] == 'failed':
                st.error(f"❌ Training failed: {job.get('error_message', 'Unknown error')}")
            
            else:
                st.info("⏳ Waiting for training to start...")
        
        with tab3:
            st.markdown("#### Full Job Details")
            st.json(job)
        
        # Actions
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if job['status'] == 'completed':
                if st.button("🎯 Make Predictions", use_container_width=True):
                    # Find corresponding model
                    models = st.session_state.api_client.list_models()
                    model = next((m for m in models if m['job_id'] == selected_job_id), None)
                    if model:
                        st.session_state.selected_model_id = model['id']
                    st.switch_page("pages/8_🎯_Make_Predictions.py")
        
        with col2:
            if st.button("💬 Ask AI About Results", use_container_width=True):
                st.switch_page("pages/7_💬_AI_Assistant.py")
        
        with col3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error loading job details: {str(e)}")

except Exception as e:
    st.error(f"❌ Error loading training jobs: {str(e)}")

