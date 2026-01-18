"""AI Assistant chat page."""
import streamlit as st
from components.api_client import APIClient
from components.auth import require_auth
from datetime import datetime

st.set_page_config(page_title="AI Assistant", page_icon="💬", layout="wide")

require_auth()

if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

# Initialize chat history
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'chat_session_id' not in st.session_state:
    st.session_state.chat_session_id = None

st.title("💬 AI Assistant")
st.markdown("Chat with the AI-powered ML assistant using LangChain")
st.markdown("---")

# Sidebar with quick actions
with st.sidebar:
    st.subheader("🚀 Quick Actions")
    
    if st.button("📊 Analyze My Dataset", use_container_width=True):
        try:
            datasets = st.session_state.api_client.list_datasets()
            if datasets:
                dataset_id = datasets[0]['id']
                st.session_state.quick_message = f"Analyze dataset with ID: {dataset_id}"
        except:
            st.error("No datasets found")
    
    if st.button("💡 Recommend a Model", use_container_width=True):
        st.session_state.quick_message = "What machine learning model should I use for my dataset?"
    
    if st.button("🩺 Diagnose Model Issues", use_container_width=True):
        try:
            models = st.session_state.api_client.list_models()
            if models:
                model_id = models[0]['id']
                st.session_state.quick_message = f"Diagnose model with ID: {model_id}"
        except:
            st.error("No models found")
    
    if st.button("🔧 Suggest Features", use_container_width=True):
        try:
            datasets = st.session_state.api_client.list_datasets()
            if datasets:
                dataset_id = datasets[0]['id']
                st.session_state.quick_message = f"Suggest features for dataset: {dataset_id}"
        except:
            st.error("No datasets found")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.chat_session_id = None
        st.rerun()

# Main chat area
st.subheader("💬 Chat")

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(message["timestamp"])

# Chat input
if 'quick_message' in st.session_state:
    prompt = st.session_state.quick_message
    del st.session_state.quick_message
else:
    prompt = st.chat_input("Ask me anything about ML, your datasets, or models...")

if prompt:
    # Add user message to chat
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.chat_messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(user_message["timestamp"])
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.api_client.send_chat_message(
                    message=prompt,
                    session_id=st.session_state.chat_session_id
                )
                
                # Update session ID
                if not st.session_state.chat_session_id:
                    st.session_state.chat_session_id = response.get('session_id')
                
                assistant_message = {
                    "role": "assistant",
                    "content": response['response'],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.chat_messages.append(assistant_message)
                
                st.markdown(response['response'])
                st.caption(assistant_message["timestamp"])
            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

# Help section
if not st.session_state.chat_messages:
    st.markdown("---")
    st.markdown("### 💡 What can I help you with?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dataset Analysis:**
        - "Analyze my dataset with ID xyz"
        - "What insights can you find in my data?"
        - "Are there any data quality issues?"
        
        **Model Selection:**
        - "What model should I use for classification?"
        - "Recommend a model for my problem"
        - "Compare Random Forest vs XGBoost"
        """)
    
    with col2:
        st.markdown("""
        **Results Interpretation:**
        - "Explain my model results"
        - "Why is my accuracy low?"
        - "What do these metrics mean?"
        
        **Feature Engineering:**
        - "Suggest features for my dataset"
        - "How can I improve my model?"
        - "What preprocessing should I do?"
        """)
    
    st.markdown("---")
    
    st.info("💡 **Tip:** The AI assistant has access to your datasets and models. You can reference them by ID or ask general ML questions!")

# Footer
st.markdown("---")
st.caption("Powered by LangChain & OpenAI")

