from fastapi import APIRouter, UploadFile, File, Depends, status, Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from io import BytesIO

from sqlalchemy.orm import Session
from typing import List

from ..database.db import get_db
from ..database.tables import Prediction
from ..utils.authorization_utils import get_current_admin_user, get_current_user, create_error_log_in_auth_service
from ..schemas.prediction_schema import PredictionResponse, TopPrediction, PredictionBase, PredictionBaseResponse
from ..utils.prediction_utils import model, CLASS_NAMES, save_image, ClassLabels

router = APIRouter()

bearer_scheme = HTTPBearer()

# predictions
@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...), 
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):  
    try:
        # Prediction logic
        # print(some_file)
        img_data = await file.read()
        img = image.load_img(BytesIO(img_data), target_size=(180, 180))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        predictions = model.predict(img)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        top_5_indices = predictions[0].argsort()[-5:][::-1]
        top_5_predictions_dict = [
            {"class_name": CLASS_NAMES[i], "confidence": float(f"{predictions[0][i]:.6f}")}
            for i in top_5_indices
        ]
        print('value of top_5_indices')
        print(top_5_indices)

        # Save to database
        db_prediction = Prediction(
            user_id=current_user["user_id"],
            image_path='image path goes here',  # Adjust as needed
            prediction={'predicted_class': predicted_class},
            top_5_prediction=top_5_predictions_dict,
            confidence=confidence,
            feedback_comment="feedback given by endpoints or edge devices"
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=float(f"{confidence:.6f}"),
            message="Prediction was successful and metadata has been saved",
            top_5_predictions=[TopPrediction(**item) for item in top_5_predictions_dict] # Convert to Pydantic models
        )
    except Exception as e:
        print('value of e')
        print(str(e))
        token = credentials.credentials
        await creating_error_log(token, current_user["user_id"], str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

async def creating_error_log(token, user_id, error):
    try:
        error_log_data = {
            "error_type": "Prediction Error",
            "error_message": f"Exception Error: {str(error)}",
            "the_model_id": None,
            "user_id": user_id
        }
        created_error_log = await create_error_log_in_auth_service(error_log_data, token)
        print('Created error log:', created_error_log)
    except HTTPException as e:
        print(f"Failed to create error log: {str(e)}")
        pass

# the following is the monitoring and logging

# Get all Predictions (Admin only)
@router.get("/predictions/", response_model=List[PredictionBaseResponse])
async def read_all_predictions(
    skip: int = 0, limit: int = 100, 
    current_user: dict = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    
    try:
        list_response = db.query(Prediction).offset(skip).limit(limit).all()
        # if len(list_response) == 0:
        #     return []
        return list_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve predictions: {str(e)}")


# Get Predictions for the current user
@router.get("/predictions/me", response_model=List[PredictionBaseResponse])
async def read_user_predictions(
    current_user: dict = Depends(get_current_user),  
    db: Session = Depends(get_db)
):
    
    try:
        list_response = db.query(Prediction).filter(Prediction.user_id == current_user["user_id"]).all()
        return list_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user predictions: {str(e)}")


# Get a specific Prediction by ID (User can access their own, Admin can access all)
@router.get("/predictions/{prediction_id}", response_model=PredictionBaseResponse)
async def read_prediction(
    prediction_id: int, 
    current_user: dict = Depends(get_current_user),  
    db: Session = Depends(get_db)
):
    
    try:
        # current_user = await get_current_user(token)
        db_prediction = db.query(Prediction).filter(Prediction.prediction_id == prediction_id).first()
        # if db_prediction is None or (db_prediction.user_id != current_user["user_id"] and not current_user["is_admin"]):
        #     raise HTTPException(status_code=404, detail="Prediction not found")
        return db_prediction
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prediction: {str(e)}")


# Delete a Prediction by ID (User can delete their own, Admin can delete all)
@router.delete("/predictions/{prediction_id}", response_model=PredictionBaseResponse)
async def delete_prediction(
    prediction_id: int, 
    current_user: dict = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    
    try:
        db_prediction = db.query(Prediction).filter(Prediction.prediction_id == prediction_id).first()
        # if db_prediction is None or (db_prediction.user_id != current_user["user_id"] and not current_user["is_admin"]):
        #     raise HTTPException(status_code=404, detail="Prediction not found")
        db.delete(db_prediction)
        db.commit()
        return db_prediction
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete prediction: {str(e)}")


