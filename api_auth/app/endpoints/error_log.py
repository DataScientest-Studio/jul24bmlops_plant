from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..schemas.error_log_schema import ErrorLogBase, ErrorLogResponse
from ..schemas.auth_schema import UserBase
from ..database.db import get_db
from ..database.tables import ErrorLog
from ..utils.auth_utils import get_current_admin_user, get_current_user

router = APIRouter()

# Create a new ErrorLog (Admin only or handled internally)
@router.post("/error_logs/", response_model=ErrorLogResponse)
def create_error_log(
    error_log: ErrorLogBase, 
    db: Session = Depends(get_db),
    current_user: UserBase = Depends(get_current_admin_user)  # Usually internal, but Admin can log manually
    ):
    try:
        db_error_log = ErrorLog(**error_log.dict(), user_id=current_user.user_id)
        db.add(db_error_log)
        db.commit()
        db.refresh(db_error_log)
        return db_error_log
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create error log: {str(e)}")

# Get all ErrorLogs (Admin only)
@router.get("/error_logs/", response_model=List[ErrorLogResponse])
def read_error_logs(
    skip: int = 0, limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: UserBase = Depends(get_current_admin_user)
    ):
    try:
        the_list_response = db.query(ErrorLog).offset(skip).limit(limit).all()
        return the_list_response
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve error logs: {str(e)}")

# Get ErrorLogs for the current user
@router.get("/error_logs/me", response_model=List[ErrorLogResponse])
def read_user_error_logs(
    db: Session = Depends(get_db),
    current_user: UserBase = Depends(get_current_user)
    ):
    try:
        the_list_response = db.query(ErrorLog).filter(ErrorLog.user_id == current_user.user_id).all()
        return the_list_response
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user error logs: {str(e)}")

# Get a specific ErrorLog by ID (User can access their own, Admin can access all)
@router.get("/error_logs/{error_id}", response_model=ErrorLogResponse)
def read_error_log(
    error_id: int, 
    db: Session = Depends(get_db),
    current_user: UserBase = Depends(get_current_user)
    ):
    try:
        db_error_log = db.query(ErrorLog).filter(ErrorLog.error_id == error_id).first()
        if db_error_log is None or (db_error_log.user_id != current_user.user_id and not current_user.is_admin):
              raise HTTPException(status_code=404, detail="ErrorLog not found")
        return db_error_log
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve error log: {str(e)}")


