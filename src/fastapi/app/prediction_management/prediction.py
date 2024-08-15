from fastapi import APIRouter, UploadFile, File, Depends
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.schemas.auth_schema import UserBase
from app.database.db import get_db
from app.auth.auth_utils import get_current_admin_user, get_current_user
from app.schemas.prediction_schema import PredictionResponse, TopPrediction, PredictionBase, PredictionBaseResponse
from app.utils.model import model
from app.utils.collect_data_utils import CLASS_NAMES, save_image, ClassLabels

router = APIRouter()

# predictions
@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
     img_data = await file.read()
     img = image.load_img(BytesIO(img_data), target_size=(180, 180))
     img = image.img_to_array(img)
     img = np.expand_dims(img, axis=0)
     
     predictions = model.predict(img)
     predicted_class_idx = np.argmax(predictions, axis=1)[0]
     predicted_class = CLASS_NAMES[predicted_class_idx]
     # new changes start from here
     confidence = predictions[0][predicted_class_idx]
     # temp 
     top_5_indices = predictions[0].argsort()[-5:][::-1]
     top_5_predictions = [
          TopPrediction(class_name=CLASS_NAMES[i], confidence=float(f"{predictions[0][i]:.6f}"))
          for i in top_5_indices
     ]
     print('top five indices')
     print(top_5_indices)
     print(top_5_predictions)
     
     if confidence >= 0.8:
          # Save image in the predicted class folder
          save_image(img_data, predicted_class, file.filename)
          message = f"Image saved with label: {predicted_class}"
     else:
          # Save image in the "other" folder
          save_image(img_data, "other", file.filename)
          message = f"Image saved in 'other' folder. Top 5 predictions: {top_5_predictions}"
     
     return PredictionResponse(
          predicted_class=predicted_class,
          confidence=float(f"{confidence:.6f}"),
          message=message,
          top_5_predictions=top_5_predictions
     )

# the following is the monitoring and logging

# Create a new PredictionBase (User or Admin)
@router.post("/predictions/", response_model=PredictionBaseResponse)
def create_prediction(
     prediction: PredictionBase, 
     db: Session = Depends(get_db),
     current_user: User = Depends(get_current_user)
     ):
     db_prediction = PredictionBase(**prediction.dict(), user_id=current_user.user_id)
     db.add(db_prediction)
     db.commit()
     db.refresh(db_prediction)
     return db_prediction

# Get all Predictions (Admin only)
@router.get("/predictions/", response_model=List[PredictionBaseResponse])
def read_predictions(
     skip: int = 0, limit: int = 100, 
     db: Session = Depends(get_db),
     current_user: User = Depends(get_current_admin_user)
     ):
     return db.query(PredictionBase).offset(skip).limit(limit).all()

# Get Predictions for the current user
@router.get("/predictions/me", response_model=List[PredictionBaseResponse])
def read_user_predictions(
     db: Session = Depends(get_db),
     current_user: User = Depends(get_current_user)
     ):
     return db.query(PredictionBase).filter(PredictionBase.user_id == current_user.user_id).all()

# Get a specific PredictionBase by ID (User can access their own, Admin can access all)
@router.get("/predictions/{prediction_id}", response_model=PredictionBaseResponse)
def read_prediction(
     prediction_id: int, 
     db: Session = Depends(get_db),
     current_user: User = Depends(get_current_user)
     ):
     db_prediction = db.query(PredictionBase).filter(PredictionBase.prediction_id == prediction_id).first()
     if db_prediction is None or (db_prediction.user_id != current_user.user_id and not current_user.is_admin):
          raise HTTPException(status_code=404, detail="PredictionBase not found")
     return db_prediction

# Delete a PredictionBase by ID (User can delete their own, Admin can delete all)
@router.delete("/predictions/{prediction_id}", response_model=PredictionBaseResponse)
def delete_prediction(
     prediction_id: int, 
     db: Session = Depends(get_db),
     current_user: User = Depends(get_current_user)
     ):
     db_prediction = db.query(PredictionBase).filter(PredictionBase.prediction_id == prediction_id).first()
     if db_prediction is None or (db_prediction.user_id != current_user.user_id and not current_user.is_admin):
          raise HTTPException(status_code=404, detail="PredictionBase not found")
     db.delete(db_prediction)
     db.commit()
     return db_prediction


