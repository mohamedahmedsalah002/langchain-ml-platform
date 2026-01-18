"""Authentication helper functions."""
import streamlit as st
from typing import Optional


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return "token" in st.session_state and "user_email" in st.session_state


def require_auth():
    """Require authentication, redirect to login if not authenticated."""
    if not is_authenticated():
        st.warning("⚠️ Please login to access this page")
        st.stop()


def logout():
    """Logout the current user."""
    if "token" in st.session_state:
        del st.session_state.token
    if "user_email" in st.session_state:
        del st.session_state.user_email
    st.success("✅ Logged out successfully!")
    st.rerun()


def get_token() -> Optional[str]:
    """Get the current user's token."""
    return st.session_state.get("token")


def get_user_email() -> Optional[str]:
    """Get the current user's email."""
    return st.session_state.get("user_email")

