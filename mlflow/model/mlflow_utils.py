from typing import Any
from datetime import datetime
import mlflow
from mlflow import MlflowClient

def dt_stamp() -> str:
  """
  creates the (date)timestamp corresponding to the current time (at the moment of the function call)
  """ 
  # Getting the current date and time
  dt = datetime.now()
  # getting the timestamp
  ts = datetime.timestamp(dt)
  # convert to datetime
  date_time = datetime.fromtimestamp(ts)
  # convert timestamp to string in dd-mm-yyyy HH:MM:SS
  str_date_time = date_time.strftime("%Y-%m-%d-%H-%M-%S")
  return str_date_time

def create_mlflow_xprmnt(experiment_name: str, tags: dict[str,Any]) -> str:
  """
  creates a new mlflow experiment with the given name and tags
  """
  try:
    experiment_id = mlflow.create_experiment(name=experiment_name,
                                             tags=tags)
  except:
    print(f"experiment {experiment_name} already exists")
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
  return experiment_id

def print_run_info(r):
  tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
  artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
  print(f"run_id: {r.info.run_id}")
  print(f"artifacts: {artifacts}")
  print(f"params: {r.data.params}")
  print(f"metrics: {r.data.metrics}")
  print(f"tags: {tags}")

def print_xprmnt_info(experiment):
  print(f"Name: {experiment.name}")
  print(f"Experiment_id: {experiment.experiment_id}")
  print(f"Artifact Location: {experiment.artifact_location}")
  print(f"Tags: {experiment.tags}")
  print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
  print(f"Creation timestamp: {experiment.creation_time}")