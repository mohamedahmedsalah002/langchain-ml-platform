"""Login page."""
import streamlit as st
from components.api_client import APIClient
from components.auth import is_authenticated

st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")

# Check if already authenticated
if is_authenticated():
    st.success("✅ You are already logged in!")
    if st.button("Go to Home"):
        st.switch_page("app.py")
    st.stop()

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

st.title("🔐 Login")
st.markdown("---")

# Login form
with st.form("login_form"):
    email = st.text_input("Email", placeholder="your.email@example.com")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login", use_container_width=True)
    
    if submit:
        if not email or not password:
            st.error("❌ Please enter both email and password")
        else:
            try:
                with st.spinner("Logging in..."):
                    response = st.session_state.api_client.login(email, password)
                    
                    # Store token and user info in session state
                    st.session_state.token = response['access_token']
                    st.session_state.user_email = email
                    
                    st.success("✅ Login successful!")
                    st.balloons()
                    
                    # Redirect to home
                    st.info("Redirecting to home...")
                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Login failed: {str(e)}")

st.markdown("---")

# Register link
col1, col2 = st.columns([2, 1])
with col1:
    st.info("Don't have an account?")
with col2:
    if st.button("Register", use_container_width=True):
        st.switch_page("pages/2_📝_Register.py")

