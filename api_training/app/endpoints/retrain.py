from fastapi import APIRouter, Depends, HTTPException, Query
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

# MLFLOW_ENDPOINT = "http://auth:8000"
MLFLOW_ENDPOINT = 'http://mlflow:8005'


# hyperparameters
class Hyperparameters(BaseModel):
    image_size: Tuple[int, int] = (180, 180)
    batch_size: int = 32
    base_learning_rate: float = 0.001
    fine_tune_at: int = 100
    initial_epochs: int = 10
    fine_tune_epochs: int = 10
    seed: int = 123
    validation_split: float = 0.2
    val_tst_split_enum: int = 1
    val_tst_split: int = 2
    chnls: Tuple[int] = (3,)
    dropout_rate: float = 0.2
    init_weights: str = "imagenet"

# training request
@router.post("/custom/train")
async def custom_train_model(
    paths: List[str] = Query(..., description="List of paths for training data"),
    params: Optional[Hyperparameters] = None
):
    print('insdie the custom trian funciton')
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MLFLOW_ENDPOINT}/train/",
                json=params.dict() if params else None,
                params={"paths": paths}
            )
        response.raise_for_status()
        print('value of response.json()')
        print(response.json())
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# retraining
@router.post("/custom/retrain")
async def custom_retrain_model(
    paths: List[str] = Query(..., description="List of paths for retraining data"),
    model_file_path: str = Query(..., description="File path for the model to retrain"),
    params: Optional[Hyperparameters] = None
):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MLFLOW_ENDPOINT}/retrain/",
                json=params.dict() if params else None,
                params={"paths": paths, "model_file_path": model_file_path}
            )
        response.raise_for_status()
        print('value of response.json()')
        print(response.json())
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))