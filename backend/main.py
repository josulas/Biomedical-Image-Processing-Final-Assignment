"""
FastAPI application for biomedical image classification inference.
"""


from datetime import datetime, timezone
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import (FastAPI,
                     HTTPException,
                     Depends,
                     File,
                     UploadFile,
                     Form)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import uvicorn
import aiofiles

from sqlite_config import init_database, get_db_connection
from models import (
    UserCreate, UserLogin, UserResponse, PatientAssignment,
    ImageResponse, StudyResultCreate,
    TokenResponse, AdminUserUpdate
)
from auth import (
    create_access_token,
    get_current_user,
    require_role,
    verify_password
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Security
security = HTTPBearer()


# File storage
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Starting up the application...")

    # Initialize database
    init_database()
    # Load model
    logger.info("APP configuration: %s", fastapi_app.openapi())
    yield
    # Shutdown
    logger.info("Shutting down the application...")


# Initialize FastAPI app
app = FastAPI(
    title="Backend API",
    description="API for serving user data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Auth endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register_user(user: UserCreate):
    """Register a new patient user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (user.email,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=400,
                detail="User with this email already exists"
            )
        # Create user (always as patient for registration)
        cursor.execute("""
            INSERT INTO users (email, password_hash, role, first_name, last_name, 
                             phone, date_of_birth, created_at)
            VALUES (?, ?, 'patient', ?, ?, ?, ?, ?)
        """, (
            user.email,
            user.password_hash,  # Frontend sends already hashed
            user.first_name,
            user.last_name,
            user.phone,
            user.date_of_birth,
            datetime.now(timezone.utc)
        ))
        user_id = cursor.lastrowid
        conn.commit()
        # Get created user
        cursor.execute("""
            SELECT id, email, role, first_name, last_name, phone, 
                   date_of_birth, created_at, is_active
            FROM users WHERE id = ?
        """, (user_id,))
        user_data = cursor.fetchone()
        return UserResponse(
            id=user_data[0],
            email=user_data[1],
            role=user_data[2],
            first_name=user_data[3],
            last_name=user_data[4],
            phone=user_data[5],
            date_of_birth=user_data[6],
            created_at=user_data[7],
            is_active=user_data[8]
        )
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Registration error: %s", str(e))
        raise HTTPException(status_code=500, detail="Registration failed") from e
    finally:
        conn.close()

@app.post("/auth/login", response_model=TokenResponse)
async def login_user(user: UserLogin):
    """Login user and return access token."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get user by email
        cursor.execute("""
            SELECT id, email, password_hash, role, first_name, last_name, is_active
            FROM users WHERE email = ?
        """, (user.email,))
        user_data = cursor.fetchone()
        if not user_data:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        user_id, email, stored_hash, role, first_name, last_name, is_active = user_data
        if not is_active:
            raise HTTPException(
                status_code=401,
                detail="Account is deactivated"
            )
        # Verify password (comparing hashes directly since frontend sends hash)
        if not verify_password(user.password_hash, stored_hash):
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        # Create access token
        token = create_access_token({"sub": str(user_id), "email": email, "role": role})
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user=UserResponse(
                id=user_id,
                email=email,
                role=role,
                first_name=first_name,
                last_name=last_name,
                is_active=is_active
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error: %s", str(e))
        raise HTTPException(status_code=500, detail="Login failed") from e
    finally:
        conn.close()

@app.put("/auth/profile")
async def update_profile(
    profile_update: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update user profile."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        update_fields = []
        values = []
        if "first_name" in profile_update:
            update_fields.append("first_name = ?")
            values.append(profile_update["first_name"])
        if "last_name" in profile_update:
            update_fields.append("last_name = ?")
            values.append(profile_update["last_name"])
        if "phone" in profile_update:
            update_fields.append("phone = ?")
            values.append(profile_update["phone"])
        if "date_of_birth" in profile_update:
            update_fields.append("date_of_birth = ?")
            values.append(profile_update["date_of_birth"])
        if update_fields:
            update_fields.append("updated_at = ?")
            values.append(datetime.now(timezone.utc))
            values.append(current_user["id"])
            cursor.execute(f"""
                UPDATE users SET {', '.join(update_fields)}
                WHERE id = ?
            """, values)
            conn.commit()
        return {"message": "Profile updated successfully"}
    except Exception as e:
        conn.rollback()
        logger.error("Profile update error: %s", str(e))
        raise HTTPException(status_code=500,
                            detail="Profile update failed") from e
    finally:
        conn.close()

@app.put("/auth/password")
async def change_password(
    password_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Change user password."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get current password hash
        cursor.execute("SELECT password_hash FROM users WHERE id = ?", (current_user["id"],))
        user_data = cursor.fetchone()
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        current_hash = user_data[0]
        # Verify current password
        if not verify_password(password_data.get("current_password", ""), current_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        # Update password
        cursor.execute("""
            UPDATE users SET password_hash = ?, updated_at = ?
            WHERE id = ?
        """, (
            password_data["new_password"],
            datetime.now(timezone.utc),
            current_user["id"]
        ))
        conn.commit()
        return {"message": "Password changed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Password change error: %s", str(e))
        raise HTTPException(status_code=500, detail="Password change failed") from e
    finally:
        conn.close()

# Patient endpoints
@app.get("/patients", response_model=list[UserResponse])
async def get_patients(current_user: dict = Depends(get_current_user)):
    """Get all patients (for neurologists and admins)."""
    require_role(current_user, ["neurologist", "admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if current_user["role"] == "neurologist":
            # Neurologists see only assigned patients
            cursor.execute("""
                SELECT DISTINCT u.id, u.email, u.role, u.first_name, u.last_name, 
                       u.phone, u.date_of_birth, u.created_at, u.is_active
                FROM users u
                JOIN patient_assignments pa ON u.id = pa.patient_id
                WHERE u.role = 'patient' AND pa.neurologist_id = ? AND pa.is_active = 1
            """, (current_user["id"],))
        else:
            # Admins see all patients
            cursor.execute("""
                SELECT id, email, role, first_name, last_name, phone, 
                       date_of_birth, created_at, is_active
                FROM users WHERE role = 'patient'
            """)
        patients = [
            UserResponse(
                id=row[0], email=row[1], role=row[2], first_name=row[3],
                last_name=row[4], phone=row[5], date_of_birth=row[6],
                created_at=row[7], is_active=row[8]
            )
            for row in cursor.fetchall()
        ]
        return patients
    finally:
        conn.close()

@app.post("/patients/{patient_id}/assign")
async def assign_patient_to_neurologist(
    patient_id: int,
    assignment: PatientAssignment,
    current_user: dict = Depends(get_current_user)
):
    """Assign a patient to a neurologist (admin only)."""
    require_role(current_user, ["admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Verify patient and neurologist exist
        cursor.execute("SELECT id FROM users WHERE id = ? AND role = 'patient'", (patient_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Patient not found")
        cursor.execute("SELECT id FROM users WHERE id = ? AND role = 'neurologist'",
                       (assignment.neurologist_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Neurologist not found")
        # Check if assignment already exists
        cursor.execute("""
            SELECT id FROM patient_assignments 
            WHERE patient_id = ? AND neurologist_id = ? AND is_active = 1
        """, (patient_id, assignment.neurologist_id))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Assignment already exists")
        # Create assignment
        cursor.execute("""
            INSERT INTO patient_assignments (patient_id, neurologist_id, assigned_by, assigned_at)
            VALUES (?, ?, ?, ?)
        """, (patient_id,
              assignment.neurologist_id,
              current_user["id"],
              datetime.now(timezone.utc)))
        conn.commit()
        return {"message": "Patient assigned successfully"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Assignment error: %s", str(e))
        raise HTTPException(status_code=500, detail="Assignment failed") from e
    finally:
        conn.close()

# Add these endpoints after existing ones

@app.get("/neurologists", response_model=list[UserResponse])
async def get_neurologists():
    """Get all active neurologists for patient assignment."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, email, role, first_name, last_name, phone, 
                   date_of_birth, created_at, is_active
            FROM users 
            WHERE role = 'neurologist' AND is_active = 1
            ORDER BY first_name, last_name
        """)
        neurologists = [
            UserResponse(
                id=row[0], email=row[1], role=row[2], first_name=row[3],
                last_name=row[4], phone=row[5], date_of_birth=row[6],
                created_at=row[7], is_active=row[8]
            )
            for row in cursor.fetchall()
        ]
        return neurologists
    finally:
        conn.close()

@app.post("/patients/self-assign")
async def self_assign_to_neurologist(
    assignment_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Patient self-assigns to neurologist."""
    require_role(current_user, ["patient"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id FROM patient_assignments 
            WHERE patient_id = ? AND neurologist_id = ? AND is_active = 1
        """, (current_user["id"], assignment_data["neurologist_id"]))
        if cursor.fetchone():
            return {"message": "Already assigned to this neurologist"}
        cursor.execute("""
            INSERT INTO patient_assignments (patient_id, neurologist_id, assigned_by, assigned_at)
            VALUES (?, ?, ?, ?)
        """, (
            current_user["id"],
            assignment_data["neurologist_id"],
            current_user["id"],  # Self-assigned
            datetime.now(timezone.utc)
        ))
        conn.commit()
        return {"message": "Self-assigned successfully"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Self-assignment error: %s", str(e))
        raise HTTPException(status_code=500, detail="Self-assignment failed") from e
    finally:
        conn.close()

@app.get("/studies/results/me")
async def get_my_study_results(current_user: dict = Depends(get_current_user)):
    """Get current user's study results."""
    require_role(current_user, ["patient"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT sr.id, sr.image_id, sr.classification_result, sr.confidence_score,
                   sr.segmentation_data, sr.notes, sr.created_at,
                   mi.original_filename, mi.study_name,
                   u.first_name, u.last_name
            FROM study_results sr
            JOIN medical_images mi ON sr.image_id = mi.id
            JOIN users u ON sr.neurologist_id = u.id
            WHERE sr.patient_id = ?
            ORDER BY sr.created_at DESC
        """, (current_user["id"],))
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "image_id": row[1],
                "classification_result": row[2],
                "confidence_score": row[3],
                "segmentation_data": row[4],
                "notes": row[5],
                "created_at": row[6],
                "image_filename": row[7],
                "study_name": row[8],
                "neurologist_name": f"{row[9]} {row[10]}"
            })
        return results
    finally:
        conn.close()

@app.delete("/images/{image_id}")
async def delete_image(
    image_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete uploaded image (if no results exist)."""
    require_role(current_user, ["patient", "admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if current_user["role"] == "patient":
            cursor.execute("""
                SELECT file_path FROM medical_images 
                WHERE id = ? AND patient_id = ?
            """, (image_id, current_user["id"]))
        else:
            cursor.execute("""
                SELECT file_path FROM medical_images 
                WHERE id = ?
            """, (image_id,))
        image_data = cursor.fetchone()
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        # Check if results exist
        cursor.execute("""
            SELECT id FROM study_results WHERE image_id = ?
        """, (image_id,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Cannot delete image with existing results")
        # Delete from database
        cursor.execute("DELETE FROM medical_images WHERE id = ?", (image_id,))
        # Delete physical file
        file_path = Path(image_data[0])
        if file_path.exists():
            file_path.unlink()
        conn.commit()
        return {"message": "Image deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Image deletion error: %s", str(e))
        raise HTTPException(status_code=500, detail="Image deletion failed") from e
    finally:
        conn.close()

@app.put("/images/{image_id}/assign")
async def update_image_assignment(
    image_id: int,
    assignment_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update image assignment (if no results exist)."""
    require_role(current_user, ["patient"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Verify image ownership
        cursor.execute("""
            SELECT id FROM medical_images 
            WHERE id = ? AND patient_id = ?
        """, (image_id, current_user["id"]))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Image not found")
        # Check if results exist
        cursor.execute("""
            SELECT id FROM study_results WHERE image_id = ?
        """, (image_id,))
        if cursor.fetchone():
            raise HTTPException(status_code=400,
                                detail="Cannot reassign image with existing results")
        # Update assignment
        cursor.execute("""
            UPDATE patient_assignments 
            SET neurologist_id = ?, assigned_at = ?
            WHERE patient_id = ? AND is_active = 1
        """, (
            assignment_data["neurologist_id"],
            datetime.now(timezone.utc),
            current_user["id"]
        ))
        conn.commit()
        return {"message": "Assignment updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Assignment update error: %s", str(e))
        raise HTTPException(status_code=500, detail="Assignment update failed") from e
    finally:
        conn.close()

# Image endpoints
@app.post("/images/upload", response_model=ImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    study_name: str = Form(""),
    current_user: dict = Depends(get_current_user)
):
    """Upload medical image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{current_user['id']}_{timestamp}_{file.filename}"
    file_path = UPLOAD_DIR / filename
    conn = get_db_connection()
    cursor = conn.cursor()
    logger.info("Uploading file: %s", file_path)
    logger.info("Current user: %s", current_user)
    logger.info("Study name: %s", study_name)
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        cursor.execute("""
            INSERT INTO medical_images (patient_id, filename, original_filename, 
                                      file_path, study_name, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            current_user["id"],
            filename,
            file.filename,
            str(file_path),
            study_name,
            datetime.utcnow()
        ))
        image_id = cursor.lastrowid
        conn.commit()
        if image_id is None:
            raise HTTPException(status_code=500, detail="Failed to save image")
        return ImageResponse(
            id=image_id,
            filename=filename,
            original_filename=file.filename or "",
            study_name=study_name,
            uploaded_at=datetime.now(timezone.utc)
        )
    except Exception as e:
        conn.rollback()
        if file_path.exists():
            file_path.unlink()
        logger.error("Upload error: %s", str(e))
        raise HTTPException(status_code=500, detail="Upload failed") from e
    finally:
        conn.close()

@app.get("/images", response_model=list[ImageResponse])
async def get_images(current_user: dict = Depends(get_current_user)):
    """Get user's medical images."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, filename, original_filename, study_name, uploaded_at
            FROM medical_images 
            WHERE patient_id = ?
            ORDER BY uploaded_at DESC
        """, (current_user["id"],))
        images = [
            ImageResponse(
                id=row[0],
                filename=row[1],
                original_filename=row[2],
                study_name=row[3],
                uploaded_at=row[4]
            )
            for row in cursor.fetchall()
        ]
        return images
    finally:
        conn.close()

# Study results endpoints
@app.post("/studies/results")
async def save_study_result(
    result: StudyResultCreate,
    current_user: dict = Depends(get_current_user)
):
    """Save study result (neurologist only)."""
    require_role(current_user, ["neurologist"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Verify neurologist has access to this patient
        cursor.execute("""
            SELECT 1 FROM patient_assignments 
            WHERE patient_id = ? AND neurologist_id = ? AND is_active = 1
        """, (result.patient_id, current_user["id"]))
        if not cursor.fetchone():
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        # Save result
        cursor.execute("""
            INSERT INTO study_results (image_id, patient_id, neurologist_id, 
                                     classification_result, confidence_score, 
                                     segmentation_data, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.image_id,
            result.patient_id,
            current_user["id"],
            result.classification_result,
            result.confidence_score,
            result.segmentation_data,
            result.notes,
            datetime.utcnow()
        ))
        conn.commit()
        return {"message": "Study result saved successfully"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Save result error: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to save result") from e
    finally:
        conn.close()

@app.get("/studies/results/{patient_id}")
async def get_study_results(
    patient_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get study results for a patient."""
    # Patients can only see their own results
    if current_user["role"] == "patient" and current_user["id"] != patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    # Neurologists need assignment
    if current_user["role"] == "neurologist":
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 1 FROM patient_assignments 
            WHERE patient_id = ? AND neurologist_id = ? AND is_active = 1
        """, (patient_id, current_user["id"]))
        if not cursor.fetchone():
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        conn.close()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT sr.id, sr.image_id, sr.classification_result, sr.confidence_score,
                   sr.segmentation_data, sr.notes, sr.created_at,
                   mi.original_filename, mi.study_name,
                   u.first_name, u.last_name
            FROM study_results sr
            JOIN medical_images mi ON sr.image_id = mi.id
            JOIN users u ON sr.neurologist_id = u.id
            WHERE sr.patient_id = ?
            ORDER BY sr.created_at DESC
        """, (patient_id,))
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "image_id": row[1],
                "classification_result": row[2],
                "confidence_score": row[3],
                "segmentation_data": row[4],
                "notes": row[5],
                "created_at": row[6],
                "image_filename": row[7],
                "study_name": row[8],
                "neurologist_name": f"{row[9]} {row[10]}"
            })
        return results
    finally:
        conn.close()

# Neurologist endpoints

@app.get("/patients/{patient_id}/images")
async def get_patient_images(
    patient_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get images for a specific patient (neurologist access)."""
    require_role(current_user, ["neurologist", "admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Verify neurologist has access to this patient
        if current_user["role"] == "neurologist":
            cursor.execute("""
                SELECT 1 FROM patient_assignments 
                WHERE patient_id = ? AND neurologist_id = ? AND is_active = 1
            """, (patient_id, current_user["id"]))
            if not cursor.fetchone():
                raise HTTPException(status_code=403, detail="Access denied to this patient")
        # Get patient's images
        cursor.execute("""
            SELECT id, filename, original_filename, study_name, uploaded_at
            FROM medical_images 
            WHERE patient_id = ?
            ORDER BY uploaded_at DESC
        """, (patient_id,))
        images = []
        for row in cursor.fetchall():
            images.append({
                "id": row[0],
                "filename": row[1],
                "original_filename": row[2],
                "study_name": row[3],
                "uploaded_at": row[4]
            })
        return images
    finally:
        conn.close()

@app.get("/images/{image_id}/results")
async def get_image_results(
    image_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get results for a specific image."""
    require_role(current_user, ["neurologist", "admin", "patient"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT sr.classification_result, sr.confidence_score, sr.notes, 
                   sr.segmentation_data, sr.created_at
            FROM study_results sr
            WHERE sr.image_id = ?
            ORDER BY sr.created_at DESC
            LIMIT 1
        """, (image_id,))
        result = cursor.fetchone()
        if result:
            return {
                "classification_result": result[0],
                "confidence_score": result[1],
                "notes": result[2],
                "segmentation_data": result[3],
                "created_at": result[4]
            }
        else:
            raise HTTPException(status_code=404, detail="No results found for this image")
    finally:
        conn.close()

@app.get("/images/{image_id}/download")
async def download_image(
    image_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Download image file."""
    require_role(current_user, ["neurologist", "admin", "patient"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT file_path, original_filename FROM medical_images 
            WHERE id = ?
        """, (image_id,))
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Image not found")
        file_path, original_filename = result
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Image file not found on disk")
        return FileResponse(
            file_path,
            media_type="image/jpeg",
            filename=original_filename
        )
    finally:
        conn.close()

# Admin endpoints
@app.get("/admin/users", response_model=list[UserResponse])
async def get_all_users(current_user: dict = Depends(get_current_user)):
    """Get all users (admin only)."""
    require_role(current_user, ["admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, email, role, first_name, last_name, phone, 
                   date_of_birth, created_at, is_active
            FROM users ORDER BY created_at DESC
        """)
        users = [
            UserResponse(
                id=row[0], email=row[1], role=row[2], first_name=row[3],
                last_name=row[4], phone=row[5], date_of_birth=row[6],
                created_at=row[7], is_active=row[8]
            )
            for row in cursor.fetchall()
        ]
        return users
    finally:
        conn.close()

@app.put("/admin/users/{user_id}")
async def update_user(
    user_id: int,
    user_update: AdminUserUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update user (admin only)."""
    require_role(current_user, ["admin"])
    # Prevent admin from editing their own account
    if user_id == current_user["id"]:
        raise HTTPException(
            status_code=403,
            detail="You cannot edit your own account through user management. \
                Use profile settings instead."
        )
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Build update query
        update_fields = []
        values = []
        if user_update.role is not None:
            update_fields.append("role = ?")
            values.append(user_update.role)
        if user_update.is_active is not None:
            update_fields.append("is_active = ?")
            values.append(user_update.is_active)
        if user_update.first_name is not None:
            update_fields.append("first_name = ?")
            values.append(user_update.first_name)
        if user_update.last_name is not None:
            update_fields.append("last_name = ?")
            values.append(user_update.last_name)
        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")
        values.append(user_id)
        cursor.execute(f"""
            UPDATE users SET {', '.join(update_fields)}
            WHERE id = ?
        """, values)
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        conn.commit()
        return {"message": "User updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Update user error: %s", str(e))
        raise HTTPException(status_code=500, detail="Update failed") from e
    finally:
        conn.close()

@app.post("/admin/users", response_model=UserResponse)
async def create_user_admin(
    user_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create user (admin only)."""
    require_role(current_user, ["admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_data["email"],))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="User with this email already exists")
        # Create user
        cursor.execute("""
            INSERT INTO users (email, password_hash, role, first_name, last_name, 
                             phone, date_of_birth, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_data["email"],
            user_data["password_hash"],
            user_data["role"],
            user_data["first_name"],
            user_data["last_name"],
            user_data.get("phone"),
            user_data.get("date_of_birth"),
            datetime.utcnow(),
            user_data.get("is_active", True)
        ))
        user_id = cursor.lastrowid
        conn.commit()
        # Return created user
        cursor.execute("""
            SELECT id, email, role, first_name, last_name, phone, 
                   date_of_birth, created_at, is_active
            FROM users WHERE id = ?
        """, (user_id,))
        user_row = cursor.fetchone()
        return UserResponse(
            id=user_row[0], email=user_row[1], role=user_row[2],
            first_name=user_row[3], last_name=user_row[4], phone=user_row[5],
            date_of_birth=user_row[6], created_at=user_row[7], is_active=user_row[8]
        )
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Admin user creation error: %s", str(e))
        raise HTTPException(status_code=500, detail="User creation failed") from e
    finally:
        conn.close()

@app.get("/admin/assignments")
async def get_assignments(current_user: dict = Depends(get_current_user)):
    """Get all patient assignments (admin only)."""
    require_role(current_user, ["admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT pa.id, pa.patient_id, pa.neurologist_id, pa.assigned_at, pa.is_active,
                   p.first_name as patient_first, p.last_name as patient_last,
                   n.first_name as neuro_first, n.last_name as neuro_last
            FROM patient_assignments pa
            JOIN users p ON pa.patient_id = p.id
            JOIN users n ON pa.neurologist_id = n.id
            ORDER BY pa.assigned_at DESC
        """)
        assignments = []
        for row in cursor.fetchall():
            assignments.append({
                "id": row[0],
                "patient_id": row[1],
                "neurologist_id": row[2],
                "assigned_at": row[3],
                "is_active": row[4],
                "patient_name": f"{row[5]} {row[6]}",
                "neurologist_name": f"{row[7]} {row[8]}"
            })
        return assignments
    finally:
        conn.close()

@app.post("/admin/assignments")
async def create_assignment_admin(
    assignment_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create patient assignment (admin only)."""
    require_role(current_user, ["admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if assignment already exists
        cursor.execute("""
            SELECT id FROM patient_assignments 
            WHERE patient_id = ? AND neurologist_id = ? AND is_active = 1
        """, (assignment_data["patient_id"], assignment_data["neurologist_id"]))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Assignment already exists")
        # Create assignment
        cursor.execute("""
            INSERT INTO patient_assignments (patient_id, neurologist_id, assigned_by, assigned_at)
            VALUES (?, ?, ?, ?)
        """, (
            assignment_data["patient_id"],
            assignment_data["neurologist_id"],
            current_user["id"],
            datetime.now(timezone.utc)
        ))
        conn.commit()
        return {"message": "Assignment created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Assignment creation error: %s", str(e))
        raise HTTPException(status_code=500, detail="Assignment creation failed") from e
    finally:
        conn.close()

@app.put("/admin/assignments/{assignment_id}")
async def update_assignment_admin(
    assignment_id: int,
    assignment_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update patient assignment (admin only)."""
    require_role(current_user, ["admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # If neurologist is being changed, delete study results
        if assignment_data.get("delete_study_results", False):
            cursor.execute("""
                DELETE FROM study_results 
                WHERE id IN (
                    SELECT sr.id FROM study_results sr
                    JOIN patient_assignments pa ON sr.patient_id = pa.patient_id
                    WHERE pa.id = ?
                )
            """, (assignment_id,))
        # Update assignment
        update_fields = []
        values = []
        if "neurologist_id" in assignment_data:
            update_fields.append("neurologist_id = ?")
            values.append(assignment_data["neurologist_id"])
        if "is_active" in assignment_data:
            update_fields.append("is_active = ?")
            values.append(assignment_data["is_active"])
        if update_fields:
            values.append(assignment_id)
            cursor.execute(f"""
                UPDATE patient_assignments SET {', '.join(update_fields)}
                WHERE id = ?
            """, values)
        conn.commit()
        return {"message": "Assignment updated successfully"}
    except Exception as e:
        conn.rollback()
        logger.error("Assignment update error: %s", str(e))
        raise HTTPException(status_code=500, detail="Assignment update failed") from e
    finally:
        conn.close()

@app.delete("/admin/assignments/{assignment_id}")
async def delete_assignment_admin(
    assignment_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete patient assignment (admin only)."""
    require_role(current_user, ["admin"])
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Delete study results for this assignment
        cursor.execute("""
            DELETE FROM study_results 
            WHERE id IN (
                SELECT sr.id FROM study_results sr
                JOIN patient_assignments pa ON sr.patient_id = pa.patient_id
                WHERE pa.id = ?
            )
        """, (assignment_id,))
        # Delete assignment
        cursor.execute("DELETE FROM patient_assignments WHERE id = ?", (assignment_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Assignment not found")
        conn.commit()
        return {"message": "Assignment deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error("Assignment deletion error: %s", str(e))
        raise HTTPException(status_code=500, detail="Assignment deletion failed") from e
    finally:
        conn.close()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy"
    )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Dementia Toolkit Backend API",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/auth/register": "User registration",
            "/auth/login": "User login",
            "/patients": "Patient management",
            "/images": "Image management",
            "/studies": "Study results",
            "/admin": "Admin functions",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
