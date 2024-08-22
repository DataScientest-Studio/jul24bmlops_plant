from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..schemas.api_request_log_schema import APIRequestLogBase, APIRequestLogResponse
from ..schemas.auth_schema import UserBase
from ..database.db import get_db
from ..database.tables import APIRequestLog
from ..auth.auth_utils import get_current_admin_user, get_current_user

router = APIRouter()

# Create a new APIRequestLog (Typically handled internally)
@router.post("/api_request_logs/", response_model=APIRequestLogResponse)
def create_api_request_log(
     api_request_log: APIRequestLogBase, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_user)
     ):
     try:
          db_api_request_log = APIRequestLog(**api_request_log.dict(), user_id=current_user.user_id)
          db.add(db_api_request_log)
          db.commit()
          db.refresh(db_api_request_log)
          return db_api_request_log
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to create api request log: {str(e)}")

# Get all APIRequestLogs (Admin only)
@router.get("/api_request_logs/", response_model=List[APIRequestLogResponse])
def read_api_request_logs(
     skip: int = 0, limit: int = 100, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_admin_user)
     ):
     try:
          the_response = db.query(APIRequestLog).offset(skip).limit(limit).all()
          return the_response
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve api request logs: {str(e)}")


# Get APIRequestLogs for the current UserBase
@router.get("/api_request_logs/me", response_model=List[APIRequestLogResponse])
def read_user_api_request_logs(
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_user)
     ):
     try:
          the_list_response = db.query(APIRequestLog).filter(APIRequestLog.user_id == current_user.user_id).all()
          return the_list_response
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve user api request logs: {str(e)}")

# Get a specific APIRequestLog by ID (UserBase can access their own, Admin can access all)
@router.get("/api_request_logs/{request_id}", response_model=APIRequestLogResponse)
def read_api_request_log(
     request_id: int, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_user)
     ):
     try:
          db_api_request_log = db.query(APIRequestLog).filter(APIRequestLog.request_id == request_id).first()
          if db_api_request_log is None or (db_api_request_log.user_id != current_user.user_id and not current_user.is_admin):
               raise HTTPException(status_code=404, detail="APIRequestLog not found")
          return db_api_request_log
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve api request logs: {str(e)}")
