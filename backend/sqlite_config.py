"""SQLite database configuration and initialization."""


import sqlite3
import os
import getpass
import hashlib
from datetime import datetime
from datetime import timezone
import logging
from pathlib import Path

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

load_dotenv()

# Database file path
DB_PATH = "/app/data/dementia_toolkit.db"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

if not ADMIN_EMAIL or not ADMIN_PASSWORD:
    raise ValueError("ADMIN_USER and ADMIN_PASSWORD environment variables must be set.")

def get_db_connection():
    """Get database connection."""
    # Ensure directory exists with proper error handling
    data_dir = Path(DB_PATH).parent
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        # Check directory permissions
        if not os.access(data_dir, os.W_OK):
            logger.error("No write permission for directory: %s", data_dir)
            raise PermissionError(f"No write permission for directory: {data_dir}")
    except PermissionError as e:
        logger.error("Permission error with data directory: %s", e)
        raise
    except Exception as e:
        logger.error("Error creating data directory: %s", e)
        raise
    try:
        # SQLite will create the database file automatically if it doesn't exist
        # No need to manually create an empty file
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError as e:
        logger.error("SQLite operational error: %s", e)
        logger.error("DB_PATH: %s", DB_PATH)
        logger.error("Directory exists: %s", data_dir.exists())
        logger.error("Directory writable: %s", os.access(data_dir, os.W_OK))
        # Additional debugging info
        try:
            stat_info = data_dir.stat()
            logger.error("Directory permissions: %s", oct(stat_info.st_mode)[-3:])
            logger.error("Directory owner UID: %s", stat_info.st_uid)
            logger.error("Current process user: %s", getpass.getuser())
        except OSError as stat_e:
            logger.error("Could not get directory stats: %s", stat_e)
        raise
    except Exception as e:
        logger.error("Unexpected database error: %s", e)
        raise

def hash_password(password: str) -> str:
    """Hash password using SHA-256. This method should be used consistently in frontend."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_database():
    """Initialize database with tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('patient', 'neurologist', 'admin')),
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                phone TEXT,
                date_of_birth DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        # Medical images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                study_name TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES users (id)
            )
        """)
        # Patient assignments table (neurologist-patient relationships)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                neurologist_id INTEGER NOT NULL,
                assigned_by INTEGER NOT NULL,
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (patient_id) REFERENCES users (id),
                FOREIGN KEY (neurologist_id) REFERENCES users (id),
                FOREIGN KEY (assigned_by) REFERENCES users (id),
                UNIQUE(patient_id, neurologist_id)
            )
        """)
        # Study results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS study_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                patient_id INTEGER NOT NULL,
                neurologist_id INTEGER NOT NULL,
                classification_result INTEGER,
                confidence_score REAL,
                segmentation_data TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES medical_images (id),
                FOREIGN KEY (patient_id) REFERENCES users (id),
                FOREIGN KEY (neurologist_id) REFERENCES users (id)
            )
        """)
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users (role)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_medical_images_patient \
                       ON medical_images (patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_assignments_patient \
                       ON patient_assignments (patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_assignments_neurologist \
                       ON patient_assignments (neurologist_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_study_results_patient ON \
                       study_results (patient_id)")
        # Create default admin user if it doesn't exist
        cursor.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1")
        if not cursor.fetchone():
            # Default admin credentials (should be changed in production)
            cursor.execute("""
                INSERT INTO users (email, password_hash, role, first_name, last_name, created_at)
                VALUES (?, ?, 'admin', 'System', 'Administrator', ?)
            """, (ADMIN_EMAIL, hash_password(ADMIN_PASSWORD), datetime.now(timezone.utc)))
            # password_hash is sha256 of "password" - should be changed immediately
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        conn.rollback()
        logger.error("Database initialization error: %s", e)
        raise
    finally:
        conn.close()
