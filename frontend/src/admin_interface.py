"""Admin interface for user and patient management."""

import streamlit as st
from datetime import date, datetime
from typing import Dict, Any, List, Optional
import hashlib

def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

class AdminInterface:
    """Interface for admin users."""
    
    def __init__(self, api_client):
        self.api_client = api_client
    
    def render(self, page: str):
        """Render admin interface based on current page."""
        if page == "admin_users":
            self.show_user_management()
        elif page == "admin_assignments":
            self.show_patient_assignments()
    
    def show_user_management(self):
        """Show user management interface."""
        st.subheader("👥 User Management")
        
        # Create tabs for create and edit
        tab1, tab2 = st.tabs(["➕ Create New User", "✏️ Edit Existing Users"])
        
        with tab1:
            self._show_create_user_form()
        
        with tab2:
            self._show_edit_users_section()
    
    def _show_create_user_form(self):
        """Show form to create new user."""
        st.markdown("#### Create New User")
        st.info("Create neurologist or admin accounts. Patients register themselves.")
        
        with st.form("create_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("First Name", placeholder="Enter first name")
                email = st.text_input("Email", placeholder="Enter email address")
                role = st.selectbox("Role", options=["neurologist", "admin"], index=0)
                password = st.text_input("Password", type="password", placeholder="Create password")
            
            with col2:
                last_name = st.text_input("Last Name", placeholder="Enter last name")
                phone = st.text_input("Phone (optional)", placeholder="Enter phone number")
                date_of_birth = st.date_input(
                    "Date of Birth (optional)",
                    value=None,
                    min_value=date(1940, 1, 1),
                    max_value=date.today()
                )
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
            
            is_active = st.checkbox("Active Account", value=True, help="User can log in")
            
            submit_create = st.form_submit_button("➕ Create User", use_container_width=True, type="primary")
            
            if submit_create:
                if all([first_name, last_name, email, password, confirm_password]):
                    if password != confirm_password:
                        st.error("❌ Passwords do not match")
                    elif len(password) < 6:
                        st.error("❌ Password must be at least 6 characters long")
                    else:
                        with st.spinner("Creating user..."):
                            user_data = {
                                "email": email,
                                "password_hash": hash_password(password),
                                "role": role,
                                "first_name": first_name,
                                "last_name": last_name,
                                "phone": phone if phone else None,
                                "date_of_birth": date_of_birth.isoformat() if date_of_birth else None,
                                "is_active": is_active
                            }
                            
                            result = self.api_client.create_user(user_data)
                            
                            if result["success"]:
                                st.success(f"✅ {role.title()} account created successfully!")
                                st.rerun()
                            else:
                                st.error(f"❌ User creation failed: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("⚠️ Please fill in all required fields")
    
    def _show_edit_users_section(self):
        """Show section to edit existing users."""
        st.markdown("#### Edit Existing Users")
        
        # Load users
        result = self.api_client.get_all_users()
        
        if not result["success"]:
            st.error(f"❌ Failed to load users: {result.get('error', 'Unknown error')}")
            return
        
        users = result["data"]
        
        if not users:
            st.info("No users found in the system.")
            return
        
        # Get current admin user ID to exclude from editing
        current_user = st.session_state.user_data
        current_admin_id = current_user['id']
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            role_filter = st.selectbox("Filter by Role", options=["All", "patient", "neurologist", "admin"], index=0)
        with col2:
            status_filter = st.selectbox("Filter by Status", options=["All", "Active", "Inactive"], index=0)
        with col3:
            search_term = st.text_input("Search by name/email", placeholder="Search...")
        
        # Filter users (exclude current admin)
        filtered_users = self._filter_users(users, role_filter, status_filter, search_term)
        # Remove current admin from the list
        filtered_users = [u for u in filtered_users if u['id'] != current_admin_id]
        
        if not filtered_users:
            st.info("No users match the current filters.")
            return
        
        # Display users in a more compact format
        for user in filtered_users:
            current_user = st.session_state.user_data
            is_current_admin = (user['id'] == current_user['id'])
            
            # Add visual indicator for current admin
            user_label = f"👤 {user['first_name']} {user['last_name']} ({user['role']}) - {'✅' if user['is_active'] else '❌'}"
            if is_current_admin:
                user_label += " 🔒 (Your Account)"
            with st.expander(f"👤 {user['first_name']} {user['last_name']} ({user['role']}) - {'✅' if user['is_active'] else '❌'}"):
                # Check if this is the current admin
                if user['id'] == current_admin_id:
                    st.warning("⚠️ You cannot edit your own account through user management. Use Account Settings instead.")
                else:
                    self._show_edit_user_form(user)

    def _filter_users(self, users: List[Dict], role_filter: str, status_filter: str, search_term: str) -> List[Dict]:
        """Filter users based on criteria."""
        filtered = users
        
        if role_filter != "All":
            filtered = [u for u in filtered if u['role'] == role_filter]
        
        if status_filter != "All":
            is_active = status_filter == "Active"
            filtered = [u for u in filtered if u['is_active'] == is_active]
        
        if search_term:
            search_lower = search_term.lower()
            filtered = [u for u in filtered if 
                       search_lower in u['first_name'].lower() or 
                       search_lower in u['last_name'].lower() or 
                       search_lower in u['email'].lower()]
        
        return filtered
    
    def _show_edit_user_form(self, user: Dict[str, Any]):
        """Show form to edit a specific user."""
        current_user = st.session_state.user_data
        if user['id'] == current_user['id']:
            st.warning("⚠️ You cannot edit your own account through user management. Use Account Settings in the sidebar instead.")
            return
        
        form_key = f"edit_user_{user['id']}"
        
        with st.form(form_key):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("First Name", value=user['first_name'], key=f"fn_{user['id']}")
                email_display = st.text_input("Email", value=user['email'], disabled=True, key=f"em_{user['id']}")
                role = st.selectbox("Role", options=["patient", "neurologist", "admin"], 
                                   index=["patient", "neurologist", "admin"].index(user['role']), 
                                   key=f"role_{user['id']}")
            
            with col2:
                last_name = st.text_input("Last Name", value=user['last_name'], key=f"ln_{user['id']}")
                phone = st.text_input("Phone", value=user.get('phone', '') or '', key=f"ph_{user['id']}")
                is_active = st.checkbox("Active Account", value=user['is_active'], key=f"active_{user['id']}")
            
            # Date of birth handling
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
                max_value=date.today(),
                key=f"dob_{user['id']}"
            )
            
            col_save, col_reset = st.columns(2)
            with col_save:
                submit_edit = st.form_submit_button("💾 Save Changes", use_container_width=True)
            with col_reset:
                if st.form_submit_button("🔄 Reset Password", use_container_width=True):
                    self._show_reset_password_form(user['id'])
            
            if submit_edit:
                if first_name and last_name:
                    with st.spinner("Updating user..."):
                        update_data = {
                            "first_name": first_name,
                            "last_name": last_name,
                            "role": role,
                            "phone": phone if phone else None,
                            "date_of_birth": date_of_birth.isoformat() if date_of_birth else None,
                            "is_active": is_active
                        }
                        
                        result = self.api_client.update_user_admin(user['id'], update_data)
                        
                        if result["success"]:
                            st.success("✅ User updated successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Update failed: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("⚠️ First name and last name are required")
    
    def _show_reset_password_form(self, user_id: int):
        """Show password reset form."""
        st.info("🔄 Password Reset - Feature to be implemented")
    
    def show_patient_assignments(self):
        """Show patient assignment interface."""
        st.subheader("🔗 Patient Assignments")
        
        # Create tabs for create and edit assignments
        tab1, tab2 = st.tabs(["➕ Create New Assignment", "✏️ Manage Assignments"])
        
        with tab1:
            self._show_create_assignment_form()
        
        with tab2:
            self._show_manage_assignments_section()
    
    def _show_create_assignment_form(self):
        """Show form to create new patient assignment."""
        st.markdown("#### Create New Assignment")
        st.info("Assign patients to neurologists for medical evaluation and analysis.")
        
        # Load patients and neurologists
        users_result = self.api_client.get_all_users()
        if not users_result["success"]:
            st.error(f"❌ Failed to load users: {users_result.get('error', 'Unknown error')}")
            return
        
        users = users_result["data"]
        patients = [u for u in users if u['role'] == 'patient']
        neurologists = [u for u in users if u['role'] == 'neurologist']
        
        if not patients:
            st.warning("⚠️ No patients found in the system.")
            return
        
        if not neurologists:
            st.warning("⚠️ No neurologists found in the system.")
            return
        
        with st.form("create_assignment_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                patient_options = [f"{p['first_name']} {p['last_name']} ({p['email']})" for p in patients]
                selected_patient_idx = st.selectbox("Select Patient", range(len(patient_options)), 
                                                   format_func=lambda x: patient_options[x])
                selected_patient = patients[selected_patient_idx]
            
            with col2:
                neurologist_options = [f"Dr. {n['first_name']} {n['last_name']} ({n['email']})" for n in neurologists]
                selected_neurologist_idx = st.selectbox("Select Neurologist", range(len(neurologist_options)), 
                                                       format_func=lambda x: neurologist_options[x])
                selected_neurologist = neurologists[selected_neurologist_idx]
            
            notes = st.text_area("Assignment Notes (optional)", placeholder="Add any special instructions or notes...")
            
            submit_assignment = st.form_submit_button("➕ Create Assignment", use_container_width=True, type="primary")
            
            if submit_assignment:
                with st.spinner("Creating assignment..."):
                    assignment_data = {
                        "patient_id": selected_patient['id'],
                        "neurologist_id": selected_neurologist['id'],
                        "notes": notes if notes else None
                    }
                    
                    result = self.api_client.create_assignment(assignment_data)
                    
                    if result["success"]:
                        st.success(f"✅ Patient {selected_patient['first_name']} {selected_patient['last_name']} assigned to Dr. {selected_neurologist['first_name']} {selected_neurologist['last_name']}!")
                        st.rerun()
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        if "already exists" in error_msg.lower():
                            st.warning("⚠️ This patient is already assigned to this neurologist.")
                        else:
                            st.error(f"❌ Assignment failed: {error_msg}")
    
    def _show_manage_assignments_section(self):
        """Show section to manage existing assignments."""
        st.markdown("#### Manage Existing Assignments")
        
        # Load assignments
        result = self.api_client.get_assignments()
        
        if not result["success"]:
            st.error(f"❌ Failed to load assignments: {result.get('error', 'Unknown error')}")
            return
        
        assignments = result["data"]
        
        if not assignments:
            st.info("No patient assignments found.")
            return
        
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Filter by Status", options=["All", "Active", "Inactive"], index=0, key="assign_filter")
        with col2:
            search_term = st.text_input("Search assignments", placeholder="Search by patient or neurologist name...", key="assign_search")
        
        # Filter assignments
        filtered_assignments = self._filter_assignments(assignments, status_filter, search_term)
        
        if not filtered_assignments:
            st.info("No assignments match the current filters.")
            return
        
        # Display assignments
        for assignment in filtered_assignments:
            status_icon = "✅" if assignment['is_active'] else "❌"
            with st.expander(f"{status_icon} {assignment['patient_name']} ↔ Dr. {assignment['neurologist_name']} (Assigned: {assignment['assigned_at'][:10]})"):
                self._show_edit_assignment_form(assignment)
    
    def _filter_assignments(self, assignments: List[Dict], status_filter: str, search_term: str) -> List[Dict]:
        """Filter assignments based on criteria."""
        filtered = assignments
        
        if status_filter != "All":
            is_active = status_filter == "Active"
            filtered = [a for a in filtered if a['is_active'] == is_active]
        
        if search_term:
            search_lower = search_term.lower()
            filtered = [a for a in filtered if 
                       search_lower in a['patient_name'].lower() or 
                       search_lower in a['neurologist_name'].lower()]
        
        return filtered
    
    def _show_edit_assignment_form(self, assignment: Dict[str, Any]):
        """Show form to edit a specific assignment."""
        form_key = f"edit_assignment_{assignment['id']}"
        
        # Load neurologists for reassignment
        users_result = self.api_client.get_all_users()
        if not users_result["success"]:
            st.error("❌ Failed to load neurologists for reassignment")
            return
        
        neurologists = [u for u in users_result["data"] if u['role'] == 'neurologist']
        
        with st.form(form_key):
            st.markdown(f"**Patient:** {assignment['patient_name']}")
            
            # Current neurologist selection
            current_neurologist_id = assignment['neurologist_id']
            neurologist_options = [f"Dr. {n['first_name']} {n['last_name']} ({n['email']})" for n in neurologists]
            
            try:
                current_idx = next(i for i, n in enumerate(neurologists) if n['id'] == current_neurologist_id)
            except StopIteration:
                current_idx = 0
            
            new_neurologist_idx = st.selectbox(
                "Assigned Neurologist",
                range(len(neurologist_options)),
                index=current_idx,
                format_func=lambda x: neurologist_options[x],
                key=f"neuro_{assignment['id']}"
            )
            new_neurologist = neurologists[new_neurologist_idx]
            
            is_active = st.checkbox("Active Assignment", value=assignment['is_active'], key=f"assign_active_{assignment['id']}")
            
            # Warning about neurologist change
            if new_neurologist['id'] != current_neurologist_id:
                st.warning("⚠️ **Warning:** Changing the neurologist will delete all diagnostic data (study results) associated with this assignment. Images will be preserved.")
            
            col_save, col_delete = st.columns(2)
            with col_save:
                submit_edit = st.form_submit_button("💾 Save Changes", use_container_width=True)
            with col_delete:
                submit_delete = st.form_submit_button("🗑️ Delete Assignment", use_container_width=True, type="secondary")
            
            if submit_edit:
                with st.spinner("Updating assignment..."):
                    update_data = {
                        "neurologist_id": new_neurologist['id'],
                        "is_active": is_active,
                        "delete_study_results": new_neurologist['id'] != current_neurologist_id
                    }
                    
                    result = self.api_client.update_assignment(assignment['id'], update_data)
                    
                    if result["success"]:
                        if new_neurologist['id'] != current_neurologist_id:
                            st.success("✅ Assignment updated and diagnostic data cleared!")
                        else:
                            st.success("✅ Assignment updated successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Update failed: {result.get('error', 'Unknown error')}")
            
            if submit_delete:
                with st.spinner("Deleting assignment..."):
                    result = self.api_client.delete_assignment(assignment['id'])
                    
                    if result["success"]:
                        st.success("✅ Assignment deleted successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Delete failed: {result.get('error', 'Unknown error')}")