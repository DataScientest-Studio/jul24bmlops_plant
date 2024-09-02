from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
# from ..utils.training_utils import train_model
# from ..schemas.retrain_schema import RetrainResponse
# from ..database.db import get_db
# from ..schemas.auth_schema import UserBase
# from ..auth.auth_utils import get_current_admin_user

router = APIRouter()

# code goes here 

# @router.post("/retrain", response_model=RetrainResponse)
# async def retrain_model(db: Session = Depends(get_db), current_user: UserBase = Depends(get_current_admin_user)):
#     try:
#         # training_data = ...  # Fetch or process your training data
#         # model = train_model(training_data)
#         # Save the model logic here
#         return {"message": "Model retrained and saved successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to retrain the model: {str(e)}")