import subprocess
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

app = FastAPI(title="Plant Recognition API")

# Define a Pydantic model for input validation
class Hyperparameters(BaseModel):
    image_size: Tuple[int, int] = Field(default=(180, 180), description="Image size (width, height)")
    batch_size: int = Field(default=32, description="Batch size")
    base_learning_rate: float = Field(default=0.001, description="Base learning rate")
    fine_tune_at: int = Field(default=100, description="Fine-tune at layer")
    initial_epochs: int = Field(default=10, description="Number of initial epochs")
    fine_tune_epochs: int = Field(default=10, description="Number of fine-tune epochs")
    seed: int = Field(default=123, description="Random seed for reproducibility")
    validation_split: float = Field(default=0.2, description="Validation split ratio")
    val_tst_split_enum: int = Field(default=1, description="Validation/test split enumeration")
    val_tst_split: int = Field(default=2, description="Validation/test split ratio")
    chnls: Tuple[int] = Field(default=(3,), description="Channels tuple")
    dropout_rate: float = Field(default=0.2, description="Dropout rate")
    init_weights: str = Field(default="imagenet", description="Initial weights for the model")

class PredictionInput(BaseModel):
    features: list

## Endpoints to train the (initial) model
# with or without hyperparameter tuning
# http://localhost:8000/train
@app.post("/train")
async def train_model(
    params: Optional[Hyperparameters] = None,
    paths: List[str] = Query(..., description="List of paths for training data"),
):
    if params:
        hyper_list = [f"{key}={value}" for key, value in params.model_dump().items()]
        cmd_lst = ["python3", "mlflow_train.py", "-i", "-p"] + hyper_list
        cmd_lst.extend(["-d"] + paths)
    else:
        cmd_lst = ["python3", "mlflow_train.py", "-i", "-d"] + paths
    subprocess.run(cmd_lst)


## Endpoints to retrain the model
# without hyperparameter tuning
# http://localhost:8000/retrain?paths=...&model_file_path=...
@app.post("/retrain")
async def retrain_model(
    params: Optional[Hyperparameters] = None,
    paths: List[str] = Query(..., description="List of paths for retraining data"),
    model_file_path: str = Query(..., description="File path for the model to retrain")
    ):
    if params:
        hyper_list = [f"{key}={value}" for key, value in params.model_dump().items()]
        cmd_lst = ["python3", "mlflow_train.py", "-p"] + hyper_list
        cmd_lst.extend(["-t"] + model_file_path)
        cmd_lst.extend(["-d"] + paths)
    else:
        cmd_lst = ["python3", "mlflow_train.py", "-t"] + model_file_path
        cmd_lst.extend(["-d"] + paths)
    subprocess.run(cmd_lst)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
