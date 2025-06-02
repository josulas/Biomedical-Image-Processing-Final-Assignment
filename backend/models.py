"""Pydantic models for request/response validation."""

from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel, EmailStr

# User models
class UserCreate(BaseModel):
    """Model for user creation."""
    email: EmailStr
    password_hash: str  # Frontend sends already hashed password
    first_name: str
    last_name: str
    phone: Optional[str] = None
    date_of_birth: Optional[date] = None

class UserLogin(BaseModel):
    """Model for user login."""
    email: EmailStr
    password_hash: str  # Frontend sends already hashed password

class UserResponse(BaseModel):
    """Model for user response."""
    id: int
    email: str
    role: str
    first_name: str
    last_name: str
    phone: Optional[str] = None
    date_of_birth: Optional[date] = None
    created_at: Optional[datetime] = None
    is_active: bool = True

class AdminUserUpdate(BaseModel):
    """Model for updating user by admin."""
    role: Optional[str] = None
    is_active: Optional[bool] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None

# Authentication models
class TokenResponse(BaseModel):
    """Model for authentication token response."""
    access_token: str
    token_type: str
    user: UserResponse

# Patient assignment models
class PatientAssignment(BaseModel):
    """Model for assigning a patient to a neurologist."""
    neurologist_id: int

# Image models
class ImageUpload(BaseModel):
    """Model for image upload."""
    study_name: Optional[str] = ""

class ImageResponse(BaseModel):
    """Model for image response."""
    id: int
    filename: str
    original_filename: str
    study_name: Optional[str] = None
    uploaded_at: datetime

# Study result models
class StudyResultCreate(BaseModel):
    """Model for creating a study result."""
    image_id: int
    patient_id: int
    classification_result: Optional[int] = None
    confidence_score: Optional[float] = None
    segmentation_data: Optional[str] = None  # JSON string
    notes: Optional[str] = None

class StudyResult(BaseModel):
    """Model for study result response."""
    id: int
    image_id: int
    patient_id: int
    neurologist_id: int
    classification_result: Optional[int] = None
    confidence_score: Optional[float] = None
    segmentation_data: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
