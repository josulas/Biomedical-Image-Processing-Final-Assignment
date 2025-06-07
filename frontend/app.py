"""Streamlit application for dementia diagnosis toolkit."""


# Python standard libraries
from datetime import datetime, date
import hashlib
import os
from dataclasses import dataclass, fields

# Third-party libraries
import streamlit as st
from dotenv import load_dotenv

# Import custom code
from src.api_client import APIClient
from src.admin_interface import AdminInterface
from src.neurologist_interface import NeurologistInterface
from src.patient_interface import PatientInterface


if not load_dotenv():
    raise RuntimeError("Failed to load environment variables from .env file")

# Configuration
API_BASE_URL = os.getenv("BACKEND_URL", "")
if not API_BASE_URL:
    raise RuntimeError("BACKEND_URL environment variable is not set in the .env file")

API_MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "")
if not API_MODEL_SERVER_URL:
    raise RuntimeError("MODEL_SERVER_URL environment variable is not set in the .env file")

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
        current_theme = st.get_option("theme.base")
        st.text(f"Current theme: {current_theme}")
        if current_theme == 'dark':
            st.session_state.selected_theme = ThemeOptions.dark
        elif current_theme == 'light':
            st.session_state.selected_theme = ThemeOptions.light
        else:
            st.session_state.selected_theme = ThemeOptions.dark
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
            theme_options = [field.default for field in fields(ThemeOptions)]
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
                apply_theme(selected_theme)
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
                "admin_users": "👥 Users",
                "admin_assignments": "🔗 Assignments"
            }
        elif user['role'] == 'neurologist':
            st.markdown("**🩺 Neurologist Tools**")
            nav_options = {
                "neurologist_patients": "👥 My Patients",
                "neurologist_analysis": "🔬 Image Analysis",
                "neurologist_assignments": "📋 Assignments"
            }
        elif user['role'] == 'patient':
            st.markdown("**📋 Patient Dashboard**")
            nav_options = {
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
            # Get real statistics
            with st.spinner("Loading dashboard statistics..."):
                # Get all users
                users_result = st.session_state.api_client.get_all_users()
                # Get all assignments
                assignments_result = st.session_state.api_client.get_assignments()
                if users_result["success"] and assignments_result["success"]:
                    users = users_result["data"]
                    assignments = assignments_result["data"]
                    # Calculate statistics
                    total_users = len(users)
                    active_patients = len([u for u in users if u['role'] == 'patient' and u['is_active']])
                    total_neurologists = len([u for u in users if u['role'] == 'neurologist'])
                    active_assignments = len([a for a in assignments if a['is_active']])
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Users", total_users)
                    with col2:
                        st.metric("Active Patients", active_patients)
                    with col3:
                        st.metric("Neurologists", total_neurologists)
                    with col4:
                        st.metric("Active Assignments", active_assignments)
                    st.divider()
                    # Recent activity section
                    st.markdown("#### Recent Activity")
                    # Show recent user registrations
                    recent_users = sorted(users, key=lambda x: x['created_at'], reverse=True)[:5]
                    if recent_users:
                        st.markdown("**Recent User Registrations:**")
                        for user in recent_users:
                            col_a, col_b, col_c = st.columns([2, 1, 1])
                            with col_a:
                                role_icon = "👤" if user['role'] == 'patient' else "🩺" if user['role'] == 'neurologist' else "⚙️"
                                st.write(f"{role_icon} **{user['first_name']} {user['last_name']}**")
                                st.caption(user['email'])
                            with col_b:
                                st.write(f"Role: {user['role'].title()}")
                                status = "✅ Active" if user['is_active'] else "❌ Inactive"
                                st.caption(status)
                            with col_c:
                                created_date = user['created_at'][:10] if user['created_at'] else "Unknown"
                                st.write(f"Joined: {created_date}")
                    st.divider()
                    # Assignment statistics
                    st.markdown("#### Assignment Overview")
                    if assignments:
                        # Group assignments by neurologist
                        neurologist_assignments = {}
                        for assignment in assignments:
                            if assignment['is_active']:
                                neuro_name = assignment['neurologist_name']
                                if neuro_name not in neurologist_assignments:
                                    neurologist_assignments[neuro_name] = 0
                                neurologist_assignments[neuro_name] += 1
                        if neurologist_assignments:
                            st.markdown("**Patient Load per Neurologist:**")
                            for neuro_name, patient_count in neurologist_assignments.items():
                                col_neuro, col_count = st.columns([3, 1])
                                with col_neuro:
                                    st.write(f"🩺 Dr. {neuro_name}")
                                with col_count:
                                    st.metric("Patients", patient_count)
                        else:
                            st.info("No active assignments found.")
                    else:
                        st.info("No patient assignments have been created yet.")
                    st.divider()
                    # Quick actions
                    st.markdown("#### Quick Actions")
                    col_action1, col_action2, col_action3 = st.columns(3)
                    with col_action1:
                        if st.button("👥 Manage Users", use_container_width=True, type="primary"):
                            st.session_state.current_page = "admin_users"
                            st.rerun()
                    with col_action2:
                        if st.button("🔗 Manage Assignments", use_container_width=True):
                            st.session_state.current_page = "admin_assignments"
                            st.rerun()
                    with col_action3:
                        unassigned_patients = len([u for u in users if u['role'] == 'patient' and u['is_active']]) - len(set([a['patient_id'] for a in assignments if a['is_active']]))
                        if st.button(f"⚠️ Unassigned Patients ({unassigned_patients})", use_container_width=True, disabled=unassigned_patients == 0):
                            st.session_state.current_page = "admin_assignments"
                            st.rerun()
                    # System health indicators
                    st.divider()
                    st.markdown("#### System Health")
                    col_health1, col_health2, col_health3 = st.columns(3)
                    with col_health1:
                        inactive_users = len([u for u in users if not u['is_active']])
                        health_color = "🟢" if inactive_users == 0 else "🟡" if inactive_users < 5 else "🔴"
                        st.metric("Inactive Users", f"{health_color} {inactive_users}")
                    with col_health2:
                        inactive_assignments = len([a for a in assignments if not a['is_active']])
                        st.metric("Inactive Assignments", inactive_assignments)
                    with col_health3:
                        # Calculate average patients per neurologist
                        if total_neurologists > 0:
                            avg_patients = round(active_assignments / total_neurologists, 1)
                            balance_status = "🟢 Balanced" if avg_patients <= 5 else "🟡 High Load" if avg_patients <= 10 else "🔴 Overloaded"
                            st.metric("Avg Patients/Neurologist", f"{avg_patients}")
                            st.caption(balance_status)
                        else:
                            st.metric("Avg Patients/Neurologist", "N/A")
                            st.caption("No neurologists")
                else:
                    st.error("❌ Failed to load dashboard statistics")
                    # Fallback metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Users", "Error")
                    with col2:
                        st.metric("Active Patients", "Error")
                    with col3:
                        st.metric("Neurologists", "Error")
                    with col4:
                        st.metric("Active Assignments", "Error")
                    st.warning("Please check your network connection and try again.")
        else:
            admin_interface = AdminInterface(st.session_state.api_client)
            admin_interface.render(st.session_state.current_page)
    elif user['role'] == 'neurologist':
        if st.session_state.current_page == "dashboard":
            st.subheader("🩺 Neurologist Dashboard")
            st.info("Welcome to your neurologist dashboard. Use the sidebar to access patient data and analysis tools.")
            # Get neurologist statistics
            patients_result = st.session_state.api_client.get_patients()
            if patients_result["success"]:
                patients = patients_result["data"]
                total_patients = len(patients)
                # Get completed studies count and pending studies
                total_completed = 0
                total_pending = 0
                for patient in patients:
                    # Get completed studies
                    results_response = st.session_state.api_client.get_study_results(patient['id'])
                    if results_response["success"]:
                        total_completed += len(results_response["data"])
                    # Get patient's images to calculate pending studies
                    images_response = st.session_state.api_client.get_patient_images(patient['id'])
                    if images_response["success"]:
                        patient_images = images_response["data"]
                        # Pending = total images - completed studies for this patient
                        patient_completed = len(results_response["data"]) if results_response["success"] else 0
                        patient_pending = len(patient_images) - patient_completed
                        total_pending += max(0, patient_pending)  # Ensure non-negative
                # Dashboard metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Assigned Patients", total_patients)
                with col2:
                    st.metric("Completed Studies", total_completed)
                with col3:
                    st.metric("Pending Studies", total_pending)
                with col4:
                    # Calculate today's reviews (studies completed today)
                    from datetime import datetime, date
                    today = date.today().isoformat()
                    todays_reviews = 0
                    for patient in patients:
                        results_response = st.session_state.api_client.get_study_results(patient['id'])
                        if results_response["success"]:
                            for result in results_response["data"]:
                                if result['created_at'].startswith(today):
                                    todays_reviews += 1
                    st.metric("Today's Reviews", todays_reviews)
                st.divider()
                # Recent activity
                st.markdown("#### Recent Activity")
                if patients:
                    # Show recent patients and their status
                    for patient in patients[:3]:  # Show last 3 patients
                        with st.container():
                            col_a, col_b, col_c = st.columns([2, 2, 1])
                            with col_a:
                                st.write(f"**{patient['first_name']} {patient['last_name']}**")
                                st.caption(patient['email'])
                            with col_b:
                                results_response = st.session_state.api_client.get_study_results(patient['id'])
                                images_response = st.session_state.api_client.get_patient_images(patient['id'])
                                if results_response["success"] and results_response["data"]:
                                    latest_result = results_response["data"][0]
                                    st.write(f"Last study: {latest_result['created_at'][:10]}")
                                    st.caption(f"Classification: {latest_result['classification_result']}")
                                elif images_response["success"] and images_response["data"]:
                                    patient_images_count = len(images_response["data"])
                                    patient_completed_count = len(results_response["data"]) if results_response["success"] else 0
                                    pending_count = patient_images_count - patient_completed_count
                                    st.write(f"Status: {pending_count} pending analysis")
                                    st.caption("Images uploaded, awaiting review")
                                else:
                                    st.write("Status: No images uploaded")
                                    st.caption("Patient needs to upload images")
                            with col_c:
                                if st.button("Analyze", key=f"quick_analyze_{patient['id']}", use_container_width=True):
                                    st.session_state.selected_patient = patient
                                    st.session_state.current_page = "neurologist_analysis"
                                    st.rerun()
                    st.divider()
                    # Quick actions
                    st.markdown("#### Quick Actions")
                    col_action1, col_action2, col_action3 = st.columns(3)
                    with col_action1:
                        if st.button("👥 View All Patients", use_container_width=True, type="primary"):
                            st.session_state.current_page = "neurologist_patients"
                            st.rerun()
                    with col_action2:
                        if st.button("🔬 Start Analysis", use_container_width=True):
                            st.session_state.current_page = "neurologist_analysis"
                            st.rerun()
                    with col_action3:
                        if st.button("📋 View Assignments", use_container_width=True):
                            st.session_state.current_page = "neurologist_assignments"
                            st.rerun()
                    # Show pending studies breakdown if there are any
                    if total_pending > 0:
                        st.divider()
                        st.markdown("#### Pending Studies Breakdown")
                        st.info(f"You have {total_pending} studies awaiting analysis across {total_patients} patients.")
                        # Show pending count per patient
                        with st.expander("📊 View Pending Studies by Patient", expanded=False):
                            for patient in patients:
                                results_response = st.session_state.api_client.get_study_results(patient['id'])
                                images_response = st.session_state.api_client.get_patient_images(patient['id'])
                                if images_response["success"]:
                                    patient_images = len(images_response["data"])
                                    patient_completed = len(results_response["data"]) if results_response["success"] else 0
                                    patient_pending = patient_images - patient_completed
                                    if patient_pending > 0:
                                        col_patient, col_pending, col_action = st.columns([2, 1, 1])
                                        with col_patient:
                                            st.write(f"**{patient['first_name']} {patient['last_name']}**")
                                        with col_pending:
                                            st.metric("Pending", patient_pending)
                                        with col_action:
                                            if st.button("Review", key=f"review_{patient['id']}", use_container_width=True):
                                                st.session_state.selected_patient = patient
                                                st.session_state.current_page = "neurologist_analysis"
                                                st.rerun()
                else:
                    st.info("You haven't been assigned any patients yet. Contact your administrator.")
            else:
                st.error(f"Failed to load dashboard data: {patients_result.get('error', 'Unknown error')}")
        else:
            neurologist_interface = NeurologistInterface(st.session_state.api_client, API_MODEL_SERVER_URL)
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
