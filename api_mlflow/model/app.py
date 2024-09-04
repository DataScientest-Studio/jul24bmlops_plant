from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional, Tuple
import uvicorn
import subprocess
import sys
from train_model import \
SEED, VALIDATION_SPLIT, VAL_TST_SPLIT_ENUM, VAL_TST_SPLIT, CHNLS, DROPOUT_RATE, INIT_WEIGHTS, \
    IMAGE_SIZE, BATCH_SIZE, BASE_LEARNING_RATE, FINE_TUNE_AT, INITIAL_EPOCHS, FINE_TUNE_EPOCHS

title = 'Training API'
pthn = 'python3'
f_path = sys.argv[0]
d_path_end = f_path.rfind('/')
d_path = sys.argv[0][:d_path_end + 1]
scrpt = d_path + 'mlflow_train.py'
i_flg = '-i'
p_flg = '-p'
d_flg = '-d'
m_flg = '-m'

app = FastAPI(title=title)

# Define a Pydantic model to mimic '-p' flag's arguments
class Hyperparameters(BaseModel):
    image_size: Optional[Tuple[int, int]] = IMAGE_SIZE
    batch_size: Optional[int] = BATCH_SIZE
    base_learning_rate: Optional[float] = BASE_LEARNING_RATE
    fine_tune_at: Optional[int] = FINE_TUNE_AT
    initial_epochs: Optional[int] = INITIAL_EPOCHS
    fine_tune_epochs: Optional[int] = FINE_TUNE_EPOCHS
    seed: Optional[int] = SEED
    validation_split: Optional[float] = VALIDATION_SPLIT
    val_tst_split_enum: Optional[int] = VAL_TST_SPLIT_ENUM
    val_tst_split: Optional[int] = VAL_TST_SPLIT
    chnls: Optional[Tuple[int]] = CHNLS
    dropout_rate: Optional[float] = DROPOUT_RATE
    init_weights: Optional[str] = INIT_WEIGHTS

## Endpoints to train the (initial) model
# without hyperparameter tuning
# http://localhost:8000/train
@app.post("/train")
async def train_model():
    cmd_lst = [pthn, scrpt, i_flg]
    subprocess.run(cmd_lst)
# with hyperparameter tuning
# http://localhost:8000/train/tune
@app.post("/train/tune")
async def train_model(params: Optional[Hyperparameters] = None):
    params_dict = dict(params)
    params_keys_lst = params_dict.keys()
    params_values_lst = params_dict.values()
    params_lst = [f"{a}={b}" for a,b in zip(params_keys_lst, params_values_lst)]
    cmd_lst = [pthn, scrpt, i_flg, p_flg] + params_lst
    subprocess.run(cmd_lst)

## Endpoints to retrain the model
# without hyperparameter tuning
# http://localhost:8000/retrain?paths=...&model_file_path=...
@app.post("/retrain")
async def retrain_model(
    paths: List[str] = Query(..., description="List of paths for retraining data"),
    model_file_path: str = Query(..., description="File path for the model to retrain")
    ):
    d_args_lst = paths
    m_arg = model_file_path
    cmd_lst = [pthn, scrpt, d_flg] + d_args_lst + [m_flg] + [m_arg]
    subprocess.run(cmd_lst)
# with hyperparameter tuning
# http://localhost:8000/retrain?paths=...&model_file_path=...
@app.post("/retrain/tune")
async def retrain_model(
    params: Optional[Hyperparameters] = None,
    paths: List[str] = Query(..., description="List of paths for retraining data"),
    model_file_path: str = Query(..., description="File path for the model to retrain")
    ):
    params_dict = dict(params)
    params_keys_lst = params_dict.keys()
    params_values_lst = params_dict.values()
    params_lst = [f"{a}={b}" for a,b in zip(params_keys_lst, params_values_lst)]
    d_args_lst = paths
    m_arg = model_file_path
    cmd_lst = [pthn, scrpt, p_flg] + params_lst + [d_flg] + d_args_lst + [m_flg] + [m_arg]
    subprocess.run(cmd_lst)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
