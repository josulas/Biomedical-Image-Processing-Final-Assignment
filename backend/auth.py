"""Authentication and authorization utilities."""

import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from sqlite_config import get_db_connection

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

security = HTTPBearer()

if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable must be set.")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password hash."""
    # Since frontend sends hashed passwords, we compare directly
    return plain_password == hashed_password

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return payload
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        ) from e

def get_current_user(token_data: dict = Depends(verify_token)):
    """Get current user from token."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, email, role, first_name, last_name, is_active
            FROM users WHERE id = ?
        """, (int(token_data["sub"]),))
        user_data = cursor.fetchone()
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        if not user_data[5]:  # is_active
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated"
            )
        return {
            "id": user_data[0],
            "email": user_data[1],
            "role": user_data[2],
            "first_name": user_data[3],
            "last_name": user_data[4]
        }
    finally:
        conn.close()

def require_role(user: dict, allowed_roles: list[str]):
    """Check if user has required role."""
    if user["role"] not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
