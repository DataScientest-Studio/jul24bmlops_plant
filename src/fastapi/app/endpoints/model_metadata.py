from fastapi import APIRouter, UploadFile, File, Depends, status, HTTPException
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import load_img, img_to_array
from app.utils.collect_data_utils import save_image, ClassLabels
import numpy as np
from io import BytesIO
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from typing import List

from app.schemas.model_metadata_schema import ModelMetadataBase, ModelMetadataResponse
from app.schemas.auth_schema import UserBase
from app.database.db import get_db
from app.database.tables import ModelMetadata
from app.auth.auth_utils import get_current_admin_user, get_current_user

router = APIRouter()


# Create ModelMetadata (Admin-only):
@router.post("/models/", response_model=ModelMetadataResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_admin_user)])
def create_model_metadata(model: ModelMetadataBase, db: Session = Depends(get_db)):
     try:
          db_model = ModelMetadata(
               model_name=model.model_name,
               version=model.version,
               training_data=model.training_data,
               training_start_time=model.training_start_time,
               training_end_time=model.training_end_time,
               accuracy=model.accuracy,
               f1_score=model.f1_score,
               precision=model.precision,
               recall=model.recall,
               training_loss=model.training_loss,
               validation_loss=model.validation_loss,
               training_accuracy=model.training_accuracy,
               validation_accuracy=model.validation_accuracy,
               training_params=model.training_params,
               logs=model.logs
          )
          db.add(db_model)
          db.commit()
          db.refresh(db_model)
          return db_model
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to create Model Meta Data: {str(e)}")

# (Admin only)
@router.get("/model/list/", response_model=List[ModelMetadataResponse])
def list_metadata(
     skip: int = 0, limit: int = 20, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_admin_user)
     ):
     try:
          list_response = db.query(ModelMetadata).offset(skip).limit(limit).all()
          return list_response
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve ModelMetadata: {str(e)}")

# Read ModelMetadata (Any authenticated user):
@router.get("/models/{model_id}", response_model=ModelMetadataResponse)
def read_model_metadata(model_id: int, db: Session = Depends(get_db), current_user: UserBase = Depends(get_current_user)):
     try:
          db_model = db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()
          if db_model is None:
               raise HTTPException(status_code=404, detail="Model not found")
          return db_model
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve Model Metadata: {str(e)}")

# Update ModelMetadata (Admin-only):
@router.put("/models/{model_id}", response_model=ModelMetadataResponse, dependencies=[Depends(get_current_admin_user)])
def update_model_metadata(model_id: int, model: ModelMetadataBase, db: Session = Depends(get_db)):
     try:
          db_model = db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()
          if db_model is None:
               raise HTTPException(status_code=404, detail="Model not found")
          for key, value in model.dict(exclude_unset=True).items():
               setattr(db_model, key, value)
          db.commit()
          db.refresh(db_model)
          return db_model
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to update Model Metadata: {str(e)}")

# Delete ModelMetadata (Admin-only):
@router.delete("/models/{model_id}", dependencies=[Depends(get_current_admin_user)])
def delete_model_metadata(model_id: int, db: Session = Depends(get_db)):
     try:
          db_model = db.query(ModelMetadata).filter(ModelMetadata.model_id == model_id).first()
          if db_model is None:
               raise HTTPException(status_code=404, detail="Model not found")
          db.delete(db_model)
          db.commit()
          return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to delete Metadata: {str(e)}")

