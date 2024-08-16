from fastapi import APIRouter, UploadFile, File, Depends, status
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
from app.database.tables import Prediction
from app.auth.auth_utils import get_current_admin_user, get_current_user
from app.schemas.prediction_schema import PredictionResponse, TopPrediction, PredictionBase, PredictionBaseResponse
from app.utils.model import model
from app.utils.collect_data_utils import CLASS_NAMES, save_image, ClassLabels

router = APIRouter()

# predictions
@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
     try:
          img_data = await file.read()
          img = image.load_img(BytesIO(img_data), target_size=(180, 180))
          img = image.img_to_array(img)
          img = np.expand_dims(img, axis=0)
          
          predictions = model.predict(img)
          predicted_class_idx = np.argmax(predictions, axis=1)[0]
          predicted_class = CLASS_NAMES[predicted_class_idx]
          # new changes start from here
          confidence = float(predictions[0][predicted_class_idx])
          # temp 
          top_5_indices = predictions[0].argsort()[-5:][::-1]
          # top_5_predictions = [
          #      TopPrediction(class_name=CLASS_NAMES[i], confidence=float(f"{predictions[0][i]:.6f}"))
          #      for i in top_5_indices
          # ]
          # save the prediction
          # top_5_predictions_dict = [prediction.dict() for prediction in top_5_predictions]
          top_5_predictions_dict = [
               {"class_name": CLASS_NAMES[i], "confidence": float(f"{predictions[0][i]:.6f}")}  
               for i in top_5_indices
          ]
          # print('value of top_5_predictions_dict')
          # print(top_5_predictions_dict)
          # temp = [
          #      {"class_label": "cat", "confidence_score": 0.85},
          #      {"class_label": "dog", "confidence_score": 0.10},
          #      {"class_label": "rabbit", "confidence_score": 0.03},
          #      {"class_label": "hamster", "confidence_score": 0.01},
          #      {"class_label": "parrot", "confidence_score": 0.01}
          # ]
          # Save to database
          db_prediction = Prediction(
               user_id=current_user.user_id,
               model_id=1,
               image_path='image path goes here',  # Adjust as needed
               prediction={'predicted_class': predicted_class},
               top_5_prediction=top_5_predictions_dict,  # Pass list of dicts
               # top_5_prediction=top_5_predictions_dict,  # Pass list of dicts
               confidence=confidence,
               feedback_comment="feedback given by endpoints or edge devices"
          )
          db.add(db_prediction)
          db.commit()
          db.refresh(db_prediction)
          
          print('saved success')
          message = 'prediction was successful and metadata has been saved'
          
          # if confidence >= 0.8:
          #      # Save image in the predicted class folder
          #      save_image(img_data, predicted_class, file.filename)
          #      message = f"Image saved with label: {predicted_class}"
          # else:
          #      # Save image in the "other" folder
          #      save_image(img_data, "other", file.filename)
          #      message = f"Image saved in 'other' folder. Top 5 predictions: {top_5_predictions}"
          
          return PredictionResponse(
               predicted_class=predicted_class,
               confidence=float(f"{confidence:.6f}"),
               message=message,
               top_5_predictions=[TopPrediction(**item) for item in top_5_predictions_dict]  # Convert to Pydantic models
          )
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# the following is the monitoring and logging

# Create a new Prediction (User or Admin)
@router.post("/predictions/", response_model=PredictionBaseResponse)
def create_prediction(
     prediction: PredictionBase, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_user)
     ):
     try:
          db_prediction = Prediction(**prediction.dict(), user_id=current_user.user_id)
          db.add(db_prediction)
          db.commit()
          db.refresh(db_prediction)
          return db_prediction
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to create prediction: {str(e)}")

# Get all Predictions (Admin only)
@router.get("/predictions/", response_model=List[PredictionBaseResponse])
def read_predictions(
     skip: int = 0, limit: int = 100, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_admin_user)
     ):
     try:
          list_response = db.query(Prediction).offset(skip).limit(limit).all()
          return list_response
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve predictions: {str(e)}")

# Get Predictions for the current user
@router.get("/predictions/me", response_model=List[PredictionBaseResponse])
def read_user_predictions(
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_user)
     ):
     try:
          list_response = db.query(Prediction).filter(Prediction.user_id == current_user.user_id).all()
          return list_response
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve user predictions: {str(e)}")

# Get a specific Prediction by ID (User can access their own, Admin can access all)
@router.get("/predictions/{prediction_id}", response_model=PredictionBaseResponse)
def read_prediction(
     prediction_id: int, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_user)
     ):
     try:
          db_prediction = db.query(Prediction).filter(Prediction.prediction_id == prediction_id).first()
          if db_prediction is None or (db_prediction.user_id != current_user.user_id and not current_user.is_admin):
               raise HTTPException(status_code=404, detail="Prediction not found")
          return db_prediction
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve prediction: {str(e)}")

# Delete a Prediction by ID (User can delete their own, Admin can delete all)
@router.delete("/predictions/{prediction_id}", response_model=PredictionBaseResponse)
def delete_prediction(
     prediction_id: int, 
     db: Session = Depends(get_db),
     current_user: UserBase = Depends(get_current_user)
     ):
     try:
          db_prediction = db.query(Prediction).filter(Prediction.prediction_id == prediction_id).first()
          if db_prediction is None or (db_prediction.user_id != current_user.user_id and not current_user.is_admin):
               raise HTTPException(status_code=404, detail="Prediction not found")
          db.delete(db_prediction)
          db.commit()
          return db_prediction
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to delete prediction: {str(e)}")


