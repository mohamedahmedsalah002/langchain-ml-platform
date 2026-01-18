"""Registration page."""
import streamlit as st
from components.api_client import APIClient
from components.auth import is_authenticated

st.set_page_config(page_title="Register", page_icon="📝", layout="centered")

# Check if already authenticated
if is_authenticated():
    st.success("✅ You are already registered and logged in!")
    if st.button("Go to Home"):
        st.switch_page("app.py")
    st.stop()

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

st.title("📝 Register")
st.markdown("---")

# Registration form
with st.form("register_form"):
    email = st.text_input("Email", placeholder="your.email@example.com")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    submit = st.form_submit_button("Register", use_container_width=True)
    
    if submit:
        if not email or not password or not confirm_password:
            st.error("❌ Please fill in all fields")
        elif password != confirm_password:
            st.error("❌ Passwords do not match")
        elif len(password) < 6:
            st.error("❌ Password must be at least 6 characters long")
        else:
            try:
                with st.spinner("Creating account..."):
                    response = st.session_state.api_client.register(email, password)
                    
                    st.success("✅ Registration successful!")
                    st.info("You can now login with your credentials")
                    
                    # Redirect to login
                    if st.button("Go to Login"):
                        st.switch_page("pages/1_🔐_Login.py")
            
            except Exception as e:
                error_msg = str(e)
                if "already registered" in error_msg.lower():
                    st.error("❌ This email is already registered")
                else:
                    st.error(f"❌ Registration failed: {error_msg}")

st.markdown("---")

# Login link
col1, col2 = st.columns([2, 1])
with col1:
    st.info("Already have an account?")
with col2:
    if st.button("Login", use_container_width=True):
        st.switch_page("pages/1_🔐_Login.py")

