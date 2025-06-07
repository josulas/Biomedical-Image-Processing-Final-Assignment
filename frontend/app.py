"""Streamlit application for dementia diagnosis toolkit."""


# Python standard libraries
from datetime import datetime, date
import hashlib
from typing import Dict, Any, Self
import os
from functools import wraps
from dataclasses import dataclass

# Third-party libraries
import requests
import streamlit as st

# Import custom code
from src.admin_interface import AdminInterface
from src.neurologist_interface import NeurologistInterface
from src.patient_interface import PatientInterface


# Configuration
API_BASE_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# Page configuration
st.set_page_config(
    page_title="Dementia Diagnosis Toolkit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class ThemeOptions:
    """Dataclass to hold theme options."""
    dark: str = "🌙 Dark"
    light: str = "☀️ Light"

class APIClient:
    """Client for backend API communication."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def set_auth_token(self, token: str):
        """Set authentication token for requests."""
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def clear_auth_token(self):
        """Clear authentication token."""
        self.session.headers.pop("Authorization", None)

    def set_token_expired(self):
        """Handle token expiration by clearing auth and triggering logout."""
        # Clear auth token
        self.clear_auth_token()
        # Set expiration flag in session state
        if hasattr(st, 'session_state'):
            st.session_state.token_expired = True

    def handle_response(self, response):
        """Handle API response."""
        response.raise_for_status()
        return {"success": True, "data": response.json()}


    @staticmethod
    def handle_token_expiration(func):
        """Decorator to handle token expiration for API methods."""
        @wraps(func)
        def wrapper(self: Self, *args, **kwargs):
            try:
                response = func(self, *args, **kwargs)
                if isinstance(response, dict) and ("success" in response or "error" in response):
                    return response
                if hasattr(response, 'status_code'):
                    return self.handle_response(response)
                return response
            except requests.exceptions.HTTPError as e:
                response = e.response
                if response.status_code == 401:
                    try:
                        error_data = response.json()
                        error_detail = error_data.get("detail", "")
                        # Check for token expiration
                        if "expired" in error_detail.lower() or "token" in error_detail.lower():
                            # Token is expired, trigger logout
                            self.set_token_expired()
                            return {"success": False,
                                    "error": "Session expired. Please login again.",
                                    "expired": True}       
                    except (ValueError, AttributeError):
                        pass
                    return {"success": False, "error": "Authentication failed", "expired": True}
                # Re-raise if not a 401 error
                return {"success": False, "error": str(e)}
            except requests.exceptions.RequestException as e:
                return {"success": False, "error": str(e)}
        return wrapper

    @handle_token_expiration
    def register_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new user."""
        response = None
        try:
            response = self.session.post(f"{self.base_url}/auth/register", json=user_data)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.HTTPError:
        # Extract error message from HTTP response
            try:
                error_detail = \
                    response.json().get("detail",
                                        "Registration failed") \
                        if response is not None else "Registration failed"
            except (ValueError, AttributeError):
                error_detail = \
                    f"HTTP {response.status_code}: Registration failed" \
                        if response is not None else "Registration failed"
            return {"success": False, "error": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def login_user(self, email: str, password_hash: str) -> Dict[str, Any]:
        """Login user and get token."""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"email": email, "password_hash": password_hash}
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def change_password(self, password_data: Dict[str, Any]) -> Dict[str, Any]:
        """Change user password."""
        try:
            response = self.session.put(f"{self.base_url}/auth/password", json=password_data)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def update_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile."""
        try:
            response = self.session.put(f"{self.base_url}/auth/profile", json=profile_data)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def get_all_users(self) -> Dict[str, Any]:
        """Get all users (admin only)."""
        try:
            response = self.session.get(f"{self.base_url}/admin/users")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user (admin only)."""
        try:
            response = self.session.post(f"{self.base_url}/admin/users", json=user_data)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def update_user_admin(self, user_id: int, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user (admin only)."""
        try:
            response = self.session.put(f"{self.base_url}/admin/users/{user_id}", json=user_data)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def get_assignments(self) -> Dict[str, Any]:
        """Get all patient assignments (admin only)."""
        try:
            response = self.session.get(f"{self.base_url}/admin/assignments")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def create_assignment(self, assignment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create patient assignment (admin only)."""
        try:
            response = self.session.post(f"{self.base_url}/admin/assignments", json=assignment_data)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def update_assignment(self,
                          assignment_id: int,
                          assignment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update patient assignment (admin only)."""
        try:
            response = \
                self.session.put(f"{self.base_url}/admin/assignments/{assignment_id}",
                                 json=assignment_data)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def delete_assignment(self, assignment_id: int) -> Dict[str, Any]:
        """Delete patient assignment (admin only)."""
        try:
            response = self.session.delete(f"{self.base_url}/admin/assignments/{assignment_id}")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def get_current_user_info(self) -> Dict[str, Any]:
        """Get current user information."""
        try:
            response = self.session.get(f"{self.base_url}/auth/me")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def get_neurologists(self) -> Dict[str, Any]:
        """Get all neurologists for assignment."""
        response = None
        try:
            response = self.session.get(f"{self.base_url}/neurologists")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.HTTPError:
            try:
                error_detail = response.json().get("detail", "Failed to load neurologists") \
                    if response is not None else "Failed to load neurologists"
            except (ValueError, AttributeError):
                error_detail = f"HTTP {response.status_code}: Failed to load neurologists" \
                    if response is not None else "Failed to load neurologists"
            return {"success": False, "error": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @handle_token_expiration
    def upload_image(self, file_data, study_name: str) -> Dict[str, Any]:
        """Upload medical image."""
        response = None
        try:
            files = {"file": file_data}
            data = {"study_name": study_name}
            response = self.session.post(f"{self.base_url}/images/upload", files=files, data=data)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.HTTPError:
            try:
                error_detail = \
                    response.json().get("detail",
                                        "Upload failed")\
                        if response is not None else "Upload failed"
            except (ValueError, AttributeError):
                error_detail = \
                    f"HTTP {response.status_code}: Upload failed"\
                        if response is not None else "Upload failed"
            return {"success": False, "error": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @handle_token_expiration
    def get_my_images(self) -> Dict[str, Any]:
        """Get current user's uploaded images."""
        try:
            response = self.session.get(f"{self.base_url}/images")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def delete_image(self, image_id: int) -> Dict[str, Any]:
        """Delete uploaded image."""
        try:
            response = self.session.delete(f"{self.base_url}/images/{image_id}")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def assign_to_neurologist(self, neurologist_id: int) -> Dict[str, Any]:
        """Self-assign to neurologist."""
        try:
            response = self.session.post(f"{self.base_url}/patients/self-assign",
                                    json={"neurologist_id": neurologist_id})
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def get_my_results(self) -> Dict[str, Any]:
        """Get current user's study results."""
        try:
            response = self.session.get(f"{self.base_url}/studies/results/me")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    @handle_token_expiration
    def update_image_assignment(self, image_id: int, neurologist_id: int) -> Dict[str, Any]:
        """Update image assignment to different neurologist."""
        try:
            response = self.session.put(f"{self.base_url}/images/{image_id}/assign",
                                    json={"neurologist_id": neurologist_id})
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

def hash_password(password: str) -> str:
    """Hash the password using a simple hash function."""
    return hashlib.sha256(password.encode()).hexdigest()

def init_session_state():
    """Initialize session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = None
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(API_BASE_URL)
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "selected_theme" not in st.session_state:
        # Get the system's theme
        current_theme = st._config.get_option('theme.base')
        if current_theme == 'dark':
            st.session_state.selected_theme = ThemeOptions.dark
        elif current_theme == 'light':
            st.session_state.selected_theme = ThemeOptions.light
        else:
            raise ValueError("Unsupported theme configuration")
    if "dashboard_tab_index" not in st.session_state:
        st.session_state.dashboard_tab_index = 0
    if "auth_checked" not in st.session_state:
        st.session_state.auth_checked = False
    if "token_expired" not in st.session_state:
        st.session_state.token_expired = False

def apply_theme(theme_selection: str):
    """Apply the selected theme using CSS injection."""
    match theme_selection:
        case ThemeOptions.dark:
            st._config.set_option('theme.base', 'dark')
        case ThemeOptions.light:
            st._config.set_option('theme.base', 'light')
        case _:
            raise ValueError("Unsupported theme selection")

def show_top_bar():
    """Display fixed top bar with language and theme options."""
    with st.container():
        _, col2, col3 = st.columns([4, 1, 1])
        with col2:
            st.selectbox(
                "🌐",
                options=["🇺🇸 English"],
                index=0,
                disabled=True,
                key="language_selector",
                help="Language selection - More languages coming soon!"
            )
        with col3:
            current_theme = st.session_state.get("selected_theme")
            theme_options = list(ThemeOptions.__dict__.values())
            try:
                current_index = theme_options.index(current_theme)
            except ValueError:
                current_index = 0
            selected_theme = st.selectbox(
                "🎨",
                options=theme_options,
                index=current_index,
                key="theme_selector",
                help="Theme selection - Choose your preferred theme"
            )
            if selected_theme != st.session_state.get("selected_theme"):
                st.session_state.selected_theme = selected_theme
                st.rerun()
    st.divider()
    if st.session_state.authenticated:
        st.markdown("## 🧠 Dementia Diagnosis Toolkit")
        st.divider()

def show_login_form():
    """Display login form."""
    show_top_bar()
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🧠 Dementia Diagnosis Toolkit")
        st.markdown("### Welcome to the Platform")
        st.info("Please sign in to continue or create a new patient account")
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        with tab1:
            st.subheader("Sign In")
            with st.form("login_form"):
                email = st.text_input(
                    "Email Address",
                    placeholder="Enter your email",
                    help="Use your registered email address"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    help="Enter your account password"
                )
                col_login, col_forgot = st.columns([2, 1])
                with col_login:
                    submit_login = st.form_submit_button(
                        "🔑 Sign In",
                        use_container_width=True,
                        type="primary"
                    )
                with col_forgot:
                    st.form_submit_button(
                        "🔒 Forgot?",
                        use_container_width=True,
                        disabled=True,
                        help="Feature coming soon"
                    )
                if submit_login:
                    if email and password:
                        with st.spinner("Signing in..."):
                            password_hash = hash_password(password)
                            result = st.session_state.api_client.login_user(email, password_hash)
                            if result["success"]:
                                st.session_state.authenticated = True
                                st.session_state.user_data = result["data"]["user"]
                                st.session_state.auth_token = result["data"]["access_token"]
                                st.session_state.api_client.set_auth_token(\
                                    result["data"]["access_token"])
                                st.success(\
                                    f"Welcome back, {st.session_state.user_data['first_name']}! 🎉")
                                st.rerun()
                            else:
                                st.error(\
                                    f"❌ Login failed: {result.get('error',
                                                                  'Invalid credentials')}")
                    else:
                        st.warning("⚠️ Please fill in all fields")
        with tab2:
            st.subheader("Create Patient Account")
            st.info("ℹ️ Only patients can register directly.\
                    Neurologists and administrators are created by system admins.")
            agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy",
                                      value=False)
            with st.form("register_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    first_name = st.text_input("First Name", placeholder="Enter your first name")
                    email_reg = st.text_input("Email", placeholder="Enter your email address")
                    password_reg = st.text_input("Password",
                                                 type="password",
                                                 placeholder="Create a secure password")
                with col_b:
                    last_name = st.text_input("Last Name",
                                              placeholder="Enter your last name")
                    phone = st.text_input("Phone (optional)",
                                          placeholder="Enter your phone number")
                    confirm_password = st.text_input("Confirm Password",
                                                     type="password",
                                                     placeholder="Confirm your password")
                date_of_birth = st.date_input(
                    "Date of Birth (optional)",
                    value=None,
                    max_value=date.today(),
                    min_value=date(1930, 1, 1),
                    help="This helps us provide age-appropriate care"
                )
                submit_register = st.form_submit_button(
                    "📝 Create Account",
                    use_container_width=True,
                    type="primary",
                    disabled=not agree_terms
                )
                if submit_register:
                    if first_name and last_name and email_reg and password_reg and confirm_password:
                        if password_reg != confirm_password:
                            st.error("❌ Passwords do not match")
                        elif len(password_reg) < 6:
                            st.error("❌ Password must be at least 6 characters long")
                        else:
                            with st.spinner("Creating your account..."):
                                user_data = {
                                    "email": email_reg,
                                    "password_hash": hash_password(password_reg),
                                    "first_name": first_name,
                                    "last_name": last_name,
                                    "phone": phone if phone else None,
                                    "date_of_birth": date_of_birth.isoformat()\
                                        if date_of_birth else None
                                }
                                result = st.session_state.api_client.register_user(user_data)
                                if result["success"]:
                                    st.success("✅ Registration successful!\
                                               Please sign in with your credentials.")
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.error(f"❌ Registration failed:\
                                             {result.get('error', 'Unknown error')}")
                    else:
                        st.warning("⚠️ Please fill in all required fields")

def show_sidebar():
    """Display compact sidebar with navigation and user info."""
    with st.sidebar:
        user = st.session_state.user_data
        # User info using native components (respects theme)
        st.markdown(f"**👤 {user['first_name']} {user['last_name']}**")
        st.caption(f"Role: {user['role'].title()}")
        st.caption(f"Email: {user['email']}")
        st.divider()
        # Account Settings - compact section
        with st.expander("🔧 Account Settings", expanded=False):
            if st.button("✏️ Edit Profile", use_container_width=True, key="edit_profile_btn"):
                st.session_state.show_edit_profile = True
                st.rerun()
            if st.button("🔒 Change Password", use_container_width=True, key="change_pass_btn"):
                st.session_state.show_change_password = True
                st.rerun()
        st.divider()
        # Role-specific navigation - compact
        if user['role'] == 'admin':
            st.markdown("**⚙️ Admin Panel**")
            nav_options = {
                "dashboard": "📋 Dashboard",
                "admin_users": "👥 Users",
                "admin_assignments": "🔗 Assignments"
            }
        elif user['role'] == 'neurologist':
            st.markdown("**🩺 Neurologist Tools**")
            nav_options = {
                "dashboard": "📋 Dashboard",
                "neurologist_patients": "👥 Patients",
                "neurologist_analysis": "🔬 Analysis"
            }
        elif user['role'] == 'patient':
            st.markdown("**📋 Patient Dashboard**")
            nav_options = {
                "dashboard": "📋 Dashboard",
                "patient_upload": "📤 Upload Images", 
                "patient_results": "📊 View Results",
                "patient_manage": "🗑️ Manage Images"
            }
        else:
            st.markdown("**🔒 Unauthorized**")
            nav_options = {
                "dashboard": "📋 Dashboard"
            }
        for page_key, page_label in nav_options.items():
            if st.button(page_label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        st.divider()
        if st.button("🏠 Dashboard", use_container_width=True, key="dashboard_home"):
            st.session_state.current_page = "dashboard"
            st.rerun()
        st.divider()
        # In the logout button handler, replace this:
        if st.button("🚪 Logout", use_container_width=True, type="secondary", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.session_state.auth_token = None
            st.session_state.api_client.clear_auth_token()
            st.session_state.auth_checked = False
            for key in list(st.session_state.keys()):
                if isinstance(key, str) and key.startswith(('show_', 'current_page')):
                    del st.session_state[key]
            st.rerun()

def show_edit_profile_modal():
    """Display edit profile form."""
    show_top_bar()
    user = st.session_state.user_data
    st.subheader("✏️ Edit Profile")
    with st.form("edit_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name", value=user.get('first_name', ''))
            _ = st.text_input("Email",
                              value=user.get('email', ''),
                              disabled=True,
                              help="Email cannot be changed")
        with col2:
            last_name = st.text_input("Last Name", value=user.get('last_name', ''))
            phone = st.text_input("Phone", value=user.get('phone', '') or '')
        if user.get('date_of_birth'):
            try:
                dob = datetime.fromisoformat(user['date_of_birth']).date()
            except ValueError:
                dob = None
        else:
            dob = None
        date_of_birth = st.date_input(
            "Date of Birth", 
            value=dob,
            min_value=date(1940, 1, 1),
            max_value=date.today()
        )
        col_a, col_b = st.columns(2)
        with col_a:
            submit_edit = st.form_submit_button("💾 Save Changes", use_container_width=True)
        with col_b:
            cancel_edit = st.form_submit_button("❌ Cancel", use_container_width=True)
        if submit_edit:
            if first_name and last_name:
                with st.spinner("Updating profile..."):
                    profile_data = {
                        "first_name": first_name,
                        "last_name": last_name,
                        "phone": phone if phone else None,
                        "date_of_birth": date_of_birth.isoformat() if date_of_birth else None
                    }
                    result = st.session_state.api_client.update_profile(profile_data)
                    if result["success"]:
                        # Update session state with new data
                        st.session_state.user_data.update({
                            "first_name": first_name,
                            "last_name": last_name,
                            "phone": phone if phone else None,
                            "date_of_birth": date_of_birth.isoformat() if date_of_birth else None
                        })
                        st.success("✅ Profile updated successfully!")
                        st.session_state.show_edit_profile = False
                        st.rerun()
                    else:
                        st.error(f"❌ Profile update failed: {result.get('error', 'Unknown error')}")
            else:
                st.warning("⚠️ First name and last name are required")
        if cancel_edit:
            st.session_state.show_edit_profile = False
            st.rerun()

def show_change_password_modal():
    """Display change password form."""
    show_top_bar()
    st.subheader("🔒 Change Password")
    with st.form("change_password_form"):
        current_password = st.text_input(
            "Current Password",
            type="password",
            placeholder="Enter current password"
        )
        new_password = st.text_input(
            "New Password",
            type="password",
            placeholder="Enter new password",
            help="Password must be at least 6 characters long"
        )
        confirm_password = st.text_input(
            "Confirm New Password",
            type="password",
            placeholder="Confirm new password"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            submit_change = \
                st.form_submit_button("🔒 Change Password", use_container_width=True)
        with col_b:
            cancel_change = \
                st.form_submit_button("❌ Cancel", use_container_width=True)
        if submit_change:
            if not all([current_password, new_password, confirm_password]):
                st.error("❌ Please fill in all fields")
            elif new_password != confirm_password:
                st.error("❌ New passwords do not match")
            elif len(new_password) < 6:
                st.error("❌ Password must be at least 6 characters long")
            elif current_password == new_password:
                st.error("❌ New password must be different from current password")
            else:
                with st.spinner("Changing password..."):
                    password_data = {
                        "current_password": hash_password(current_password),
                        "new_password": hash_password(new_password)
                    }
                    result = st.session_state.api_client.change_password(password_data)
                    if result["success"]:
                        st.success("✅ Password changed successfully!")
                        st.session_state.show_change_password = False
                        st.rerun()
                    else:
                        st.error(f"❌ Password change failed: {result.get('error',
                                                                         'Unknown error')}")
        if cancel_change:
            st.session_state.show_change_password = False
            st.rerun()
def main():
    """Main application logic."""
    init_session_state()
    apply_theme(st.session_state.selected_theme)
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"
    if "show_edit_profile" not in st.session_state:
        st.session_state.show_edit_profile = False
    if "show_change_password" not in st.session_state:
        st.session_state.show_change_password = False
    if not st.session_state.authenticated:
        show_login_form()
        return
    show_sidebar()
    if st.session_state.show_edit_profile:
        show_edit_profile_modal()
        return
    if st.session_state.show_change_password:
        show_change_password_modal()
        return
    show_top_bar()
    user = st.session_state.user_data
    if st.session_state.current_page == "dashboard":
        st.title(f"Welcome, {user['first_name']}!")
    if user['role'] == 'admin':
        if st.session_state.current_page == "dashboard":
            st.subheader("⚙️ Administrator Dashboard")
            st.info("Welcome to the admin panel. Use the sidebar to manage users and assignments.")
            # Quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Users", "Loading...")
            with col2:
                st.metric("Active Patients", "Loading...")
            with col3:
                st.metric("Neurologists", "Loading...")
        else:
            admin_interface = AdminInterface(st.session_state.api_client)
            admin_interface.render(st.session_state.current_page)
    elif user['role'] == 'neurologist':
        if st.session_state.current_page == "dashboard":
            st.subheader("🩺 Neurologist Dashboard")
            st.info("Welcome to your neurologist dashboard.\
                    Use the sidebar to access patient data and analysis tools.")
        else:
            neurologist_interface = NeurologistInterface(st.session_state.api_client)
            neurologist_interface.render(st.session_state.current_page)
    elif user['role'] == 'patient':
        if st.session_state.current_page == "dashboard":
            st.subheader("📋 Patient Dashboard")
            st.info("Welcome to your patient dashboard.\
                    You can upload medical images and view your results.")
            images_result = st.session_state.api_client.get_my_images()
            results_result = st.session_state.api_client.get_my_results()
            if images_result["success"] and results_result["success"]:
                images = images_result["data"]
                results = results_result["data"]
                total_images = len(images)
                completed_studies = len(results)
                pending_studies = total_images - completed_studies
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Uploaded Images",
                              total_images,
                              help="Total number of medical images uploaded")
                with col2:
                    st.metric("Pending Studies",
                              pending_studies,
                              help="Studies awaiting neurologist evaluation")
                with col3:
                    st.metric("Completed Studies",
                              completed_studies,
                              help="Studies with completed analysis")
                with col4:
                    if results:
                        latest_result_date = results[0]['created_at'][:10]
                        st.metric("Latest Result",
                                  latest_result_date,
                                  help="Date of most recent completed study")
                    else:
                        st.metric("Latest Result",
                                  "None",
                                  help="No completed studies yet")
                st.divider()
                st.markdown("#### Recent Activity")
                recent_items = []
                for img in images[:3]:
                    recent_items.append({
                        "type": "upload",
                        "date": img['uploaded_at'],
                        "title": f"📤 Uploaded: {img['study_name'] or img['original_filename']}",
                        "status": "⏳ Pending" \
                            if not any(r['image_id'] == img['id'] \
                                       for r in results) else "✅ Completed"
                    })
                for result in results[:3]:
                    recent_items.append({
                        "type": "result",
                        "date": result['created_at'],
                        "title": f"📊 Analysis Complete: {result['study_name'] \
                                                          or result['image_filename']}",
                        "status": f"👨‍⚕️ Dr. {result['neurologist_name']}"
                    })
                recent_items.sort(key=lambda x: x['date'], reverse=True)
                if recent_items:
                    for item in recent_items[:5]:
                        col_a, col_b, col_c = st.columns([3, 2, 1])
                        with col_a:
                            st.write(item['title'])
                        with col_b:
                            st.caption(item['status'])
                        with col_c:
                            st.caption(item['date'][:10])
                else:
                    st.info("No recent activity. Upload your first medical image to get started!")
                st.divider()
                st.markdown("#### Quick Actions")
                col_action1, col_action2, col_action3 = st.columns(3)
                with col_action1:
                    if st.button("📤 Upload New Image",
                                 use_container_width=True,
                                 type="primary"):
                        st.session_state.current_page = "patient_upload"
                        st.rerun()
                with col_action2:
                    if st.button("📊 View My Results",
                                 use_container_width=True,
                                 disabled=completed_studies == 0):
                        st.session_state.current_page = "patient_results"
                        st.rerun()
                with col_action3:
                    if st.button("🔧 Manage Images",
                                 use_container_width=True,
                                 disabled=total_images == 0):
                        st.session_state.current_page = "patient_manage"
                        st.rerun()
            else:
                st.error("❌ Failed to load dashboard data")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Uploaded Images", "Error")
                with col2:
                    st.metric("Completed Studies", "Error")
        else:
            patient_interface = PatientInterface(st.session_state.api_client)
            patient_interface.render(st.session_state.current_page)

if __name__ == "__main__":
    main()
