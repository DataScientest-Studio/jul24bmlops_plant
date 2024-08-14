from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.schemas.error_log_schema import ErrorLogBase, ErrorLogResponse
from app.schemas.auth_schema import UserBase
from app.database.db import get_db
from app.auth.auth_utils import get_current_admin_user, get_current_user

router = APIRouter()

# Create a new ErrorLogBase (Admin only or handled internally)
@router.post("/error_logs/", response_model=ErrorLogResponse)
def create_error_log(
     error_log: ErrorLogBase, 
     db: Session = Depends(get_db),
     current_user: User = Depends(get_current_admin_user)  # Usually internal, but Admin can log manually
     ):
     db_error_log = ErrorLogBase(**error_log.dict(), user_id=current_user.user_id)
     db.add(db_error_log)
     db.commit()
     db.refresh(db_error_log)
     return db_error_log

# Get all ErrorLogs (Admin only)
@router.get("/error_logs/", response_model=List[ErrorLogResponse])
def read_error_logs(
     skip: int = 0, limit: int = 100, 
     db: Session = Depends(get_db),
     current_user: User = Depends(get_current_admin_user)
     ):
     return db.query(ErrorLogBase).offset(skip).limit(limit).all()

# Get ErrorLogs for the current user
@router.get("/error_logs/me", response_model=List[ErrorLogResponse])
def read_user_error_logs(
     db: Session = Depends(get_db),
     current_user: User = Depends(get_current_user)
     ):
     return db.query(ErrorLogBase).filter(ErrorLogBase.user_id == current_user.user_id).all()

# Get a specific ErrorLogBase by ID (User can access their own, Admin can access all)
@router.get("/error_logs/{error_id}", response_model=ErrorLogResponse)
def read_error_log(
     error_id: int, 
     db: Session = Depends(get_db),
     current_user: User = Depends(get_current_user)
     ):
     db_error_log = db.query(ErrorLogBase).filter(ErrorLogBase.error_id == error_id).first()
     if db_error_log is None or (db_error_log.user_id != current_user.user_id and not current_user.is_admin):
          raise HTTPException(status_code=404, detail="ErrorLogBase not found")
     return db_error_log
