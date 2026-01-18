"""Predictions page."""
import streamlit as st
from components.api_client import APIClient
from components.auth import require_auth
import pandas as pd
import json

st.set_page_config(page_title="Make Predictions", page_icon="🎯", layout="wide")

require_auth()

if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

st.title("🎯 Make Predictions")
st.markdown("Use your trained models to make predictions")
st.markdown("---")

# Fetch trained models
try:
    models = st.session_state.api_client.list_models()
    
    if not models:
        st.info("📭 No trained models found. Train a model first.")
        if st.button("Train a Model"):
            st.switch_page("pages/5_🤖_Train_Model.py")
        st.stop()
    
    # Model selection
    st.subheader("1️⃣ Select Model")
    
    model_options = {
        f"{m['model_type'].replace('_', ' ').title()} - {m['id'][:8]}... (Acc: {m['metrics'].get('accuracy', m['metrics'].get('r2', 0)):.2%})": m['id']
        for m in models
    }
    
    # Pre-select if coming from results page
    default_idx = 0
    if 'selected_model_id' in st.session_state:
        for idx, (name, model_id) in enumerate(model_options.items()):
            if model_id == st.session_state.selected_model_id:
                default_idx = idx
                break
    
    selected_model_name = st.selectbox("Choose a model", list(model_options.keys()), index=default_idx)
    selected_model_id = model_options[selected_model_name]
    
    # Get model details
    selected_model = next(m for m in models if m['id'] == selected_model_id)
    
    # Display model info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", selected_model['model_type'].replace('_', ' ').title())
    with col2:
        st.metric("Problem Type", selected_model['problem_type'].capitalize())
    with col3:
        st.metric("Target", selected_model['target_column'])
    with col4:
        metric_value = selected_model['metrics'].get('accuracy', selected_model['metrics'].get('r2', 0))
        metric_name = "Accuracy" if 'accuracy' in selected_model['metrics'] else "R²"
        st.metric(metric_name, f"{metric_value:.2%}")
    
    st.markdown("---")
    
    # Prediction input
    st.subheader("2️⃣ Input Data")
    
    feature_columns = selected_model['feature_columns']
    
    # Choose input method
    input_method = st.radio("Input Method", ["Manual Input", "Upload File"], horizontal=True)
    
    if input_method == "Manual Input":
        st.markdown("#### Enter values for each feature:")
        
        # Create input form
        input_data = {}
        
        # Arrange inputs in columns
        num_cols = 3
        cols = st.columns(num_cols)
        
        for idx, feature in enumerate(feature_columns):
            with cols[idx % num_cols]:
                # Try to infer input type (simplified)
                input_data[feature] = st.text_input(
                    feature,
                    placeholder="Enter value",
                    key=f"input_{feature}"
                )
        
        st.markdown("---")
        
        # Make prediction button
        if st.button("🎯 Make Prediction", use_container_width=True, type="primary"):
            # Validate inputs
            if not all(input_data.values()):
                st.error("❌ Please fill in all features")
            else:
                try:
                    # Convert numeric values
                    processed_input = {}
                    for key, value in input_data.items():
                        try:
                            # Try to convert to float
                            processed_input[key] = float(value)
                        except ValueError:
                            # Keep as string
                            processed_input[key] = value
                    
                    with st.spinner("Making prediction..."):
                        result = st.session_state.api_client.predict(
                            model_id=selected_model_id,
                            input_data=processed_input
                        )
                        
                        st.success("✅ Prediction Complete!")
                        
                        # Display results
                        st.markdown("### 📊 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Prediction")
                            st.markdown(f"# {result['prediction']}")
                        
                        with col2:
                            if result.get('probabilities'):
                                st.markdown("#### Confidence")
                                
                                # Create bar chart for probabilities
                                import plotly.graph_objects as go
                                
                                probs = result['probabilities']
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=list(probs.keys()),
                                        y=list(probs.values()),
                                        marker_color='lightblue'
                                    )
                                ])
                                fig.update_layout(
                                    title="Class Probabilities",
                                    xaxis_title="Class",
                                    yaxis_title="Probability",
                                    yaxis_range=[0, 1],
                                    height=300
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed results
                        with st.expander("📋 Detailed Results"):
                            st.json(result)
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
    
    else:  # Upload File
        st.markdown("#### Upload a file with multiple rows for batch predictions")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="File should contain columns matching the model's features"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown("##### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check if all required features are present
                missing_features = set(feature_columns) - set(df.columns)
                if missing_features:
                    st.error(f"❌ Missing required features: {missing_features}")
                else:
                    st.success(f"✅ All {len(feature_columns)} required features found!")
                    
                    if st.button("🎯 Make Batch Predictions", use_container_width=True, type="primary"):
                        try:
                            with st.spinner(f"Making predictions for {len(df)} rows..."):
                                predictions = []
                                
                                # Make predictions for each row
                                progress_bar = st.progress(0)
                                
                                for idx, row in df.iterrows():
                                    input_data = row[feature_columns].to_dict()
                                    
                                    result = st.session_state.api_client.predict(
                                        model_id=selected_model_id,
                                        input_data=input_data
                                    )
                                    
                                    predictions.append(result['prediction'])
                                    progress_bar.progress((idx + 1) / len(df))
                                
                                # Add predictions to dataframe
                                df['prediction'] = predictions
                                
                                st.success(f"✅ Completed {len(predictions)} predictions!")
                                
                                # Display results
                                st.markdown("### 📊 Prediction Results")
                                st.dataframe(df, use_container_width=True)
                                
                                # Download button
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name=f"predictions_{selected_model_id[:8]}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Batch prediction failed: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")

