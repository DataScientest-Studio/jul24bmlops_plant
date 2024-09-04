from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..schemas.retrain_schema import ABTestingResultBase, ABTestingResultResponse
from ..database.db import get_db
from ..database.tables import ABTestingResult
from ..utils.authorization_utils import get_current_admin_user, get_token_from_request

router = APIRouter()

# Create a new AB Testing result (Admin only)
@router.post("/ab_testing_results/", response_model=ABTestingResultResponse)
async def create_ab_testing_result(
     ab_testing_result: ABTestingResultBase, 
     db: Session = Depends(get_db)
     # token: str = Depends(get_token_from_request)
     ):
     try:
          # current_admin_user = await get_current_admin_user(token)
          db_ab_testing_result = ABTestingResult(**ab_testing_result.dict())
          db.add(db_ab_testing_result)
          db.commit()
          db.refresh(db_ab_testing_result)
          return db_ab_testing_result
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to create AB testing: {str(e)}")

# Get all AB Testing results (Admin only)
@router.get("/ab_testing_results/", response_model=List[ABTestingResultResponse])
async def read_ab_testing_results(
     skip: int = 0, limit: int = 100, 
     db: Session = Depends(get_db)
     # token: str = Depends(get_token_from_request)
     ):
     try:
          # current_admin_user = await get_current_admin_user(token)
          the_response = db.query(ABTestingResult).offset(skip).limit(limit).all()
          return the_response
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve AB testing results: {str(e)}")


# Get a specific AB Testing result by ID (Admin only)
@router.get("/ab_testing_results/{test_id}", response_model=ABTestingResultResponse)
async def read_ab_testing_result(
     test_id: int, 
     db: Session = Depends(get_db)
     # token: str = Depends(get_token_from_request)
     ):
     try:
          # current_admin_user = await get_current_admin_user(token)
          db_ab_testing_result = db.query(ABTestingResult).filter(ABTestingResult.test_id == test_id).first()
          if db_ab_testing_result is None:
               raise HTTPException(status_code=404, detail="AB Testing Result not found")
          return db_ab_testing_result
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve AB testing results: {str(e)}")

# Delete an AB Testing result by ID (Admin only)
@router.delete("/ab_testing_results/{test_id}", response_model=ABTestingResultResponse)
async def delete_ab_testing_result(
     test_id: int, 
     db: Session = Depends(get_db)
     # token: str = Depends(get_token_from_request)
     ):
     try:
          # current_admin_user = await get_current_admin_user(token)
          db_ab_testing_result = db.query(ABTestingResult).filter(ABTestingResult.test_id == test_id).first()
          if db_ab_testing_result is None:
               raise HTTPException(status_code=404, detail="AB Testing Result not found")
          db.delete(db_ab_testing_result)
          db.commit()
          return db_ab_testing_result
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to delete AB testing results: {str(e)}")
