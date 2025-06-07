"""
API Client for Streamlit Application
This module provides a client for communicating with the backend API.
"""


from functools import wraps
from typing import Any, Dict, Self

import requests
import streamlit as st


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

    @handle_token_expiration
    def get_patients(self) -> Dict[str, Any]:
        """Get patients assigned to current neurologist or all patients (admin)."""
        try:
            response = self.session.get(f"{self.base_url}/patients")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = e.response.json().get("detail", "Failed to load patients") \
                    if e.response is not None else "Failed to load patients"
            except (ValueError, AttributeError):
                error_detail = f"HTTP {e.response.status_code}: Failed to load patients" \
                    if e.response is not None else "Failed to load patients"
            return {"success": False, "error": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @handle_token_expiration
    def get_study_results(self, patient_id: int) -> Dict[str, Any]:
        """Get study results for a specific patient."""
        try:
            response = self.session.get(f"{self.base_url}/studies/results/{patient_id}")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = e.response.json().get("detail", "Failed to load study results") \
                    if e.response is not None else "Failed to load study results"
            except (ValueError, AttributeError):
                error_detail = f"HTTP {e.response.status_code}: Failed to load study results" \
                    if e.response is not None else "Failed to load study results"
            return {"success": False, "error": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @handle_token_expiration
    def get_patient_images(self, patient_id: int) -> Dict[str, Any]:
        """Get images for a specific patient."""
        try:
            response = self.session.get(f"{self.base_url}/patients/{patient_id}/images")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = e.response.json().get("detail", "Failed to load patient images") \
                    if e.response is not None else "Failed to load patient images"
            except (ValueError, AttributeError):
                error_detail = f"HTTP {e.response.status_code}: Failed to load patient images" \
                    if e.response is not None else "Failed to load patient images"
            return {"success": False, "error": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}
