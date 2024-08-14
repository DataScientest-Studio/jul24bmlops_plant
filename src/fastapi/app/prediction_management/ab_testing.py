from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.schemas.ab_testing_schema import ABTestingResultBase, ABTestingResultResponse
from app.schemas.auth_schema import UserBase
from app.database.db import get_db
from app.auth.auth_utils import get_current_admin_user, get_current_user

router = APIRouter()

# Create a new AB Testing result (Admin only)
@router.post("/ab_testing_results/", response_model=ABTestingResultResponse)
def create_ab_testing_result(
     ab_testing_result: ABTestingResultBase, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_admin_user)
     ):
     db_ab_testing_result = ABTestingResultBase(**ab_testing_result.dict())
     db.add(db_ab_testing_result)
     db.commit()
     db.refresh(db_ab_testing_result)
     return db_ab_testing_result

# Get all AB Testing results (Admin only)
@router.get("/ab_testing_results/", response_model=List[ABTestingResultResponse])
def read_ab_testing_results(
     skip: int = 0, limit: int = 100, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_admin_user)
     ):
     return db.query(ABTestingResultBase).offset(skip).limit(limit).all()

# Get a specific AB Testing result by ID (Admin only)
@router.get("/ab_testing_results/{test_id}", response_model=ABTestingResultResponse)
def read_ab_testing_result(
     test_id: int, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_admin_user)
     ):
     db_ab_testing_result = db.query(ABTestingResultBase).filter(ABTestingResultBase.test_id == test_id).first()
     if db_ab_testing_result is None:
          raise HTTPException(status_code=404, detail="AB Testing Result not found")
     return db_ab_testing_result

# Delete an AB Testing result by ID (Admin only)
@router.delete("/ab_testing_results/{test_id}", response_model=ABTestingResultResponse)
def delete_ab_testing_result(
     test_id: int, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_admin_user)
     ):
     db_ab_testing_result = db.query(ABTestingResultBase).filter(ABTestingResultBase.test_id == test_id).first()
     if db_ab_testing_result is None:
          raise HTTPException(status_code=404, detail="AB Testing Result not found")
     db.delete(db_ab_testing_result)
     db.commit()
     return db_ab_testing_result
