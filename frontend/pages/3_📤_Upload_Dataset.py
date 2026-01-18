"""Dataset upload page."""
import streamlit as st
from components.api_client import APIClient
from components.auth import require_auth
import pandas as pd

st.set_page_config(page_title="Upload Dataset", page_icon="📤", layout="wide")

require_auth()

if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

st.title("📤 Upload Dataset")
st.markdown("Upload your dataset to start training machine learning models")
st.markdown("---")

# File upload
st.subheader("Select a File")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
    help="Supported formats: CSV, Excel, JSON, Parquet"
)

if uploaded_file is not None:
    # Show file details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Filename", uploaded_file.name)
    with col2:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.metric("File Size", f"{file_size_mb:.2f} MB")
    with col3:
        file_ext = uploaded_file.name.split('.')[-1].upper()
        st.metric("Format", file_ext)
    
    st.markdown("---")
    
    # Preview data
    st.subheader("📊 Data Preview")
    
    try:
        # Read file for preview
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        
        # Reset file pointer for upload
        uploaded_file.seek(0)
        
        # Show preview
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # Upload button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 Upload Dataset", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Uploading dataset..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Upload to backend
                        response = st.session_state.api_client.upload_dataset(uploaded_file)
                        
                        st.success("✅ Dataset uploaded successfully!")
                        st.balloons()
                        
                        # Show dataset info
                        st.json(response)
                        
                        # Option to view uploaded datasets
                        if st.button("View My Datasets"):
                            st.switch_page("pages/4_🔍_Explore_Data.py")
                
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("Please ensure the file is in a valid format")

else:
    st.info("👆 Please upload a dataset file to get started")
    
    st.markdown("### 📋 Supported File Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **CSV** (.csv) - Comma-separated values
        - **Excel** (.xlsx, .xls) - Microsoft Excel
        """)
    
    with col2:
        st.markdown("""
        - **JSON** (.json) - JavaScript Object Notation
        - **Parquet** (.parquet) - Apache Parquet
        """)
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Maximum file size: 100 MB
    - Ensure your dataset has column headers
    - Remove any sensitive or personal information
    - Check for missing values before uploading
    """)

