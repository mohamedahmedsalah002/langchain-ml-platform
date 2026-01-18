"""Dashboard page."""
import streamlit as st
from components.api_client import APIClient
from components.auth import require_auth
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

require_auth()

if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

st.title("📈 Dashboard")
st.markdown("Overview of your ML platform activity")
st.markdown("---")

try:
    # Fetch all data
    datasets = st.session_state.api_client.list_datasets()
    models = st.session_state.api_client.list_models()
    jobs = st.session_state.api_client.list_training_jobs()
    
    # Summary metrics
    st.subheader("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📁 Datasets",
            len(datasets),
            delta=f"{len([d for d in datasets if d['status'] == 'ready'])} ready"
        )
    
    with col2:
        st.metric(
            "🤖 Models",
            len(models),
            delta=f"{len([m for m in models if m['metrics'].get('accuracy', m['metrics'].get('r2', 0)) > 0.8])} high-performing"
        )
    
    with col3:
        active_jobs = len([j for j in jobs if j['status'] in ['pending', 'running']])
        st.metric(
            "⚡ Active Jobs",
            active_jobs
        )
    
    with col4:
        completed_jobs = len([j for j in jobs if j['status'] == 'completed'])
        total_jobs = len(jobs)
        success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        st.metric(
            "✅ Success Rate",
            f"{success_rate:.1f}%",
            delta=f"{completed_jobs}/{total_jobs} jobs"
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Job Status Distribution")
        
        if jobs:
            status_counts = {}
            for job in jobs:
                status = job['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    hole=0.4
                )
            ])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training jobs yet")
    
    with col2:
        st.subheader("🤖 Models by Type")
        
        if models:
            model_counts = {}
            for model in models:
                model_type = model['model_type'].replace('_', ' ').title()
                model_counts[model_type] = model_counts.get(model_type, 0) + 1
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(model_counts.keys()),
                    y=list(model_counts.values()),
                    marker_color='lightblue'
                )
            ])
            fig.update_layout(
                height=300,
                xaxis_title="Model Type",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No models trained yet")
    
    st.markdown("---")
    
    # Model performance comparison
    if models:
        st.subheader("🎯 Model Performance Comparison")
        
        model_performance = []
        for model in models:
            metric_value = model['metrics'].get('accuracy', model['metrics'].get('r2', 0))
            metric_name = "Accuracy" if 'accuracy' in model['metrics'] else "R²"
            
            model_performance.append({
                'Model Type': model['model_type'].replace('_', ' ').title(),
                'ID': model['id'][:8] + '...',
                'Problem': model['problem_type'].capitalize(),
                metric_name: f"{metric_value:.2%}",
                'Score': metric_value
            })
        
        df = pd.DataFrame(model_performance)
        
        # Sort by score
        df = df.sort_values('Score', ascending=False)
        
        # Display table (without Score column)
        display_df = df.drop('Score', axis=1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Performance chart
        fig = px.bar(
            df,
            x='Model Type',
            y='Score',
            color='Problem',
            title="Model Performance Scores",
            labels={'Score': 'Performance Score'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent activity
    st.subheader("📋 Recent Activity")
    
    if jobs:
        # Get recent jobs (last 5)
        recent_jobs = sorted(jobs, key=lambda x: x.get('started_at', ''), reverse=True)[:5]
        
        activity_data = []
        for job in recent_jobs:
            status_emoji = {
                'completed': '✅',
                'running': '🔄',
                'pending': '⏳',
                'failed': '❌'
            }
            
            activity_data.append({
                'Model': job['model_type'].replace('_', ' ').title(),
                'Status': f"{status_emoji.get(job['status'], '❓')} {job['status'].capitalize()}",
                'Target': job['target_column'],
                'Started': job.get('started_at', 'N/A')[:19] if job.get('started_at') else 'N/A'
            })
        
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, use_container_width=True, hide_index=True)
    else:
        st.info("No recent activity")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📤 Upload Dataset", use_container_width=True):
            st.switch_page("pages/3_📤_Upload_Dataset.py")
    
    with col2:
        if st.button("🤖 Train Model", use_container_width=True):
            st.switch_page("pages/5_🤖_Train_Model.py")
    
    with col3:
        if st.button("🎯 Make Predictions", use_container_width=True):
            st.switch_page("pages/8_🎯_Make_Predictions.py")
    
    with col4:
        if st.button("💬 AI Assistant", use_container_width=True):
            st.switch_page("pages/7_💬_AI_Assistant.py")

except Exception as e:
    st.error(f"❌ Error loading dashboard data: {str(e)}")
    st.info("Please ensure the backend is running and you're connected")

