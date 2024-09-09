from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx 
import os
from typing import List, Optional, Tuple
from ..utils.authorization_utils import get_current_admin_user, create_error_log_in_auth_service
from ..schemas.retrain_schema import Hyperparameters

from sqlalchemy.orm import Session

router = APIRouter()

bearer_scheme = HTTPBearer()


MLFLOW_ENDPOINT = os.getenv("MLFLOW_ENDPOINT", "")


# training request
@router.post("/custom/train")
async def custom_train_model(
    paths: List[str] = Query(..., description="List of paths for training data"),
    params: Optional[Hyperparameters] = None,
    current_user: dict = Depends(get_current_admin_user),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    print('insdie the custom trian funciton')
    print(f"MLFLOW_ENDPOINT: {MLFLOW_ENDPOINT}")
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
        token = credentials.credentials
        await creating_error_log(token, current_user['user_id'], str(e))
        print('value of e')
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# retraining
@router.post("/custom/retrain")
async def custom_retrain_model(
    paths: List[str] = Query(..., description="List of paths for retraining data"),
    model_file_path: str = Query(..., description="File path for the model to retrain"),
    params: Optional[Hyperparameters] = None,
    current_user: dict = Depends(get_current_admin_user),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
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
        token = credentials.credentials
        await creating_error_log(token, current_user['user_id'], str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def creating_error_log(token, user_id, error):
    try:
        error_log_data = {
            "error_type": "(Re)Training Error",
            "error_message": f"Exception Error: {str(error)}",
            "the_model_id": None,
            "user_id": user_id
        }
        created_error_log = await create_error_log_in_auth_service(error_log_data, token)
        print('Created error log:', created_error_log)
    except HTTPException as e:
        print(f"Failed to create error log: {str(e)}")
        pass