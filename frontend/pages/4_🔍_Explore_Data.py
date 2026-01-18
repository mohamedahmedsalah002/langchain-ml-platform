"""Data exploration page."""
import streamlit as st
from components.api_client import APIClient
from components.auth import require_auth
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Explore Data", page_icon="🔍", layout="wide")

require_auth()

if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

st.title("🔍 Explore Data")
st.markdown("Analyze and visualize your datasets")
st.markdown("---")

# Fetch datasets
try:
    datasets = st.session_state.api_client.list_datasets()
    
    if not datasets:
        st.info("📭 No datasets found. Please upload a dataset first.")
        if st.button("Upload Dataset"):
            st.switch_page("pages/3_📤_Upload_Dataset.py")
        st.stop()
    
    # Dataset selection
    dataset_options = {f"{ds['filename']} (ID: {ds['id'][:8]}...)": ds['id'] for ds in datasets}
    selected_name = st.selectbox("Select a dataset", list(dataset_options.keys()))
    selected_id = dataset_options[selected_name]
    
    # Get selected dataset details
    selected_dataset = next(ds for ds in datasets if ds['id'] == selected_id)
    
    st.markdown("---")
    
    # Dataset information
    st.subheader("📊 Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{selected_dataset.get('num_rows', 0):,}")
    with col2:
        st.metric("Columns", selected_dataset.get('num_columns', 0))
    with col3:
        size_mb = selected_dataset.get('size', 0) / (1024 * 1024)
        st.metric("Size", f"{size_mb:.2f} MB")
    with col4:
        status = selected_dataset.get('status', 'unknown')
        st.metric("Status", status.capitalize())
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Preview", "📊 Statistics", "📈 Visualizations"])
    
    with tab1:
        st.subheader("Data Preview")
        
        # Preview controls
        num_rows = st.slider("Number of rows to display", 5, 100, 10)
        
        try:
            preview_data = st.session_state.api_client.preview_dataset(selected_id, rows=num_rows)
            
            df = pd.DataFrame(preview_data['data'])
            st.dataframe(df, use_container_width=True)
            
            st.info(f"Showing {preview_data['rows_shown']} of {preview_data['total_rows']} rows")
        
        except Exception as e:
            st.error(f"Error loading preview: {str(e)}")
    
    with tab2:
        st.subheader("Dataset Statistics")
        
        try:
            profile = st.session_state.api_client.profile_dataset(selected_id)
            
            # Column information
            st.markdown("#### Column Details")
            
            col_info_data = []
            for col, info in profile['column_info'].items():
                col_info_data.append({
                    'Column': col,
                    'Type': info['dtype'],
                    'Unique Values': info['unique_values'],
                    'Missing': info['missing_count'],
                    'Missing %': f"{info['missing_percentage']:.2f}%"
                })
            
            col_df = pd.DataFrame(col_info_data)
            st.dataframe(col_df, use_container_width=True)
            
            # Detailed statistics
            st.markdown("#### Detailed Statistics")
            
            for col, stats in profile['statistics'].items():
                with st.expander(f"📊 {col}"):
                    if isinstance(stats, dict):
                        if 'mean' in stats:
                            # Numerical statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{stats.get('mean', 0):.2f}")
                            with col2:
                                st.metric("Std Dev", f"{stats.get('std', 0):.2f}")
                            with col3:
                                st.metric("Min", f"{stats.get('min', 0):.2f}")
                            with col4:
                                st.metric("Max", f"{stats.get('max', 0):.2f}")
                        
                        elif 'top_values' in stats:
                            # Categorical statistics
                            st.markdown(f"**Unique Values:** {stats.get('unique_count', 0)}")
                            st.markdown("**Top Values:**")
                            for value, count in stats['top_values'].items():
                                st.write(f"- {value}: {count}")
        
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
    
    with tab3:
        st.subheader("Data Visualizations")
        
        try:
            # Get preview data for visualization
            preview_data = st.session_state.api_client.preview_dataset(selected_id, rows=100)
            df = pd.DataFrame(preview_data['data'])
            
            # Column selection for visualization
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols:
                st.markdown("#### Numerical Features Distribution")
                
                selected_numeric = st.selectbox("Select numerical column", numeric_cols)
                
                if selected_numeric:
                    fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
                    st.plotly_chart(fig, use_container_width=True)
            
            if categorical_cols:
                st.markdown("#### Categorical Features")
                
                selected_categorical = st.selectbox("Select categorical column", categorical_cols)
                
                if selected_categorical:
                    value_counts = df[selected_categorical].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f"Top 10 values in {selected_categorical}",
                               labels={'x': selected_categorical, 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
            
            if len(numeric_cols) >= 2:
                st.markdown("#### Correlation Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key='x')
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key='y')
                
                if x_col and y_col:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")
    
    st.markdown("---")
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Train Model", use_container_width=True):
            st.session_state.selected_dataset_id = selected_id
            st.switch_page("pages/5_🤖_Train_Model.py")
    
    with col2:
        if st.button("💬 Ask AI Assistant", use_container_width=True):
            st.session_state.selected_dataset_id = selected_id
            st.switch_page("pages/7_💬_AI_Assistant.py")
    
    with col3:
        if st.button("🗑️ Delete Dataset", use_container_width=True, type="secondary"):
            if st.checkbox("Confirm deletion"):
                try:
                    st.session_state.api_client.delete_dataset(selected_id)
                    st.success("✅ Dataset deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error deleting dataset: {str(e)}")

except Exception as e:
    st.error(f"❌ Error loading datasets: {str(e)}")
    st.info("Please check your connection to the backend")

