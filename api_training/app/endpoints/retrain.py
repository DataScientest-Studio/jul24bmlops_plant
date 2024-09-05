from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import httpx  # An HTTP client library to make requests
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session
# from ..utils.training_utils import train_model
# from ..schemas.retrain_schema import RetrainResponse
# from ..database.db import get_db
# from ..schemas.auth_schema import UserBase
# from ..auth.auth_utils import get_current_admin_user

router = APIRouter()

# code goes here 

MLFLOW_ENDPOINT = 'http://mlflow:8000'


# hyperparameters
class Hyperparameters(BaseModel):
    image_size: Optional[Tuple[int, int]] = None
    batch_size: Optional[int] = None
    base_learning_rate: Optional[float] = None
    fine_tune_at: Optional[int] = None
    initial_epochs: Optional[int] = None
    fine_tune_epochs: Optional[int] = None
    seed: Optional[int] = None
    validation_split: Optional[float] = None
    val_tst_split_enum: Optional[int] = None
    val_tst_split: Optional[int] = None
    chnls: Optional[Tuple[int]] = None
    dropout_rate: Optional[float] = None
    init_weights: Optional[str] = None


# without hyperparameter tuning
@router.post("/trigger-train")
async def trigger_train():
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{MLFLOW_ENDPOINT}/train")
    return response.json()

# sample inputs: 
# Hyperparameters

# Endpoint to trigger training with hyperparameter tuning
@router.post("/trigger-train-tune")
async def trigger_train_tune(params: Hyperparameters):
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{MLFLOW_ENDPOINT}/train/tune", json=params.dict())
    return response.json()

# sample input for trigger_retrian:
# paths=["/data/retrain1", "/data/retrain2"]
# model_file_path="/models/initial_model.h5"

# Endpoint to trigger retraining without hyperparameter tuning
@router.post("/trigger-retrain")
async def trigger_retrain(paths: List[str], model_file_path: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MLFLOW_ENDPOINT}/retrain",
            params={"paths": paths, "model_file_path": model_file_path}
        )
    return response.json()

# sample inputs:
# Hyperparameters
# paths=["/data/retrain1", "/data/retrain2"]
# model_file_path="/models/initial_model.h5"

# Endpoint to trigger retraining with hyperparameter tuning
@router.post("/trigger-retrain-tune")
async def trigger_retrain_tune(params: Hyperparameters, paths: List[str], model_file_path: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MLFLOW_ENDPOINT}/retrain/tune",
            json=params.dict(),
            params={"paths": paths, "model_file_path": model_file_path}
        )
    return response.json()
