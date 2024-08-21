from typing import Any
from datetime import datetime
import mlflow

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

def create_mlflow_xprmnt(experiment_name: str, artifact_location: str, tags: dict[str,Any]) -> str:
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