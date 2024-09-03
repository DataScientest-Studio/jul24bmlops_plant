from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import mlflow
from train_model import TrainPR

app = FastAPI()

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

# Endpoint to train the model
# http://localhost:8000/train?paths=
@app.post("/train")
async def train_model(
    params: Optional[Hyperparameters] = None,
    paths: Optional[List[str]] = Query(None, description="List of paths for training data"),
    model_file_path: str = Query(..., description="File path for the model to retrain")):
    try:
        # Instance of the Model
        train_and_log_model(params, paths, model_file_path)
        return {"status": "Model trained and logged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to make predictions using a logged model
@app.post("/predict")
async def predict(input: PredictionInput, model_name: str = "random_forest_model", stage: str = "None"):
    pass
    # try:
        # Instance of the Model
        # model = load_model(model_name, stage)
        # prediction = model.predict([input.features])
        # return {"prediction": prediction.tolist()}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

# Endpoint to fetch model information (e.g., metrics, parameters)
@app.get("/model_info/{model_name}")
async def model_info(model_name: str, stage: str = "None"):
    pass
    # try:
        # Instance of the Model
        # model_uri = f"models:/{model_name}/{stage}"
        # model_info = mlflow.get_run(model_uri)
        # return {"model_info": model_info.data.to_dictionary()}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

def train_and_log_model(params=None, paths=None, model_file_path=None):
    print(f"Parameters: {params}")
    print(f"Paths: {paths}")
    print(f"Model File Path: {model_file_path}")
    with mlflow.start_run(run_name="test_GG", tags={"dev": "LM"}) as run:
        print("Debug 1")
        train_pr = TrainPR()
        print("Debug 2")
        train_pr.load_data(model_file_path)
        print("Debug 3")


if __name__ == "__main__":
    ## sets the default location for the 'mlruns' directory which represents the default local storage location for MLflow entities and artifacts 
    # one of the ways to launch a web interface that displays run data stored in the 'mlruns' directory is the command line 'mlflow ui --backend-store-uri <MLFLOW_TRACK_DIR_PATH>'
    mlflow.set_tracking_uri("/Volumes/data/Projects/py/model_test/data/training")

    ## logs metrics, parameters, and models without the need for explicit log statements 
    # logs model signatures (describing model inputs and outputs), trained models (as MLflow model artifacts) & dataset information to the active fluent run
    mlflow.autolog()

    experiment_name = "test_GG"
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name, tags={"dev": "LM"})
    except:
        print(f"experiment test_GG already exists")
        experiment_id = mlflow.get_experiment_by_name("test_GG").experiment_id
    
    mlflow.set_experiment(experiment_name)

    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
