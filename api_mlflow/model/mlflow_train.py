"""
script that performs the (re)training of the (target) model thus producing a new ([re]trained) model based on the output conditions of 'retrain_decider.py'
"""

import argparse
import sys
import os
from datetime import datetime

import mlflow
from predict_model import prdct, show_clsf_rprt, show_conf_mtrx
from train_model import TrainPR


def cli_parameters():
    parser = argparse.ArgumentParser(
        description="Train a Plant Recognition ML model with MLflow"
    )

    # Optional arguments
    parser.add_argument(
        "-d",
        "--data",
        nargs='+',
        help="Path to the data (required for both initialization and training)",
    )
    parser.add_argument(
        "-i",
        "--init",
        action="store_true",
        help="Initialize the model with the provided data",
    )
    parser.add_argument(
        "-p",
        "--hyper",
        nargs='+',
        help="hyperparameter tuning (for either training or retraining - possible for both)",
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        help="Path to load the trained model (requires data path)",
    )

    args = parser.parse_args()

    # Define all allowed combinations
    allowed_combinations = [
        (0, 1, 1, 1),
        (0, 1, 0, 1),
        (1, 0, 1, 1),
        (1, 0, 0, 1)
    ]

    init_val = 1 if args.init else 0  # True -> 1, False -> 0
    train_val = 1 if args.train is not None else 0  # Non-None -> 1, None -> 0
    hyper_val = 1 if args.hyper else 0  # True -> 1, False -> 0
    data_val = 1 if args.data is not None else 0  # Non-None -> 1, None -> 0

    # Validate argument combinations
    if (init_val, train_val, hyper_val, data_val) not in allowed_combinations:
        print("Error: invalid combination of parameters.")
        sys.exit(1)

    return args


def create_run_timestamp():
    # Get the current date and time
    now = datetime.now()
    
    # Format the date and time as RUN_YYMMDD_HH_MM_SEC
    formatted_time = now.strftime("RUN_%y%m%d_%H_%M_%S")
    
    return formatted_time


def create_mlflow_xprmnt(experiment_name: str, tags: dict[str, any]) -> str:
    """
    creates a new mlflow experiment with the given name and tags
    """
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name, tags=tags)
    except:
        print(f"experiment {experiment_name} already exists")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    return experiment_id


def parse_hyper_list(hyper_list):
    hyper_dict = {}
    for item in hyper_list:
        key, value = item.split("=")
        # Handle tuple values and cast them appropriately
        try:
            value = eval(value)  # Convert string representations of tuples or other types to their proper Python types
        except:
            pass  # If eval fails (e.g., for a string), just keep the value as is

        hyper_dict[key] = value
    return hyper_dict


def first_training(hyper, data):
    if hyper:
        train_pr = TrainPR(**parse_hyper_list(hyper))
        mlflow.log_param(hyper)
    else:
        train_pr = TrainPR()

    # Use os.path.join to combine the model_images_dir with each element in data
    data = [os.path.join(model_images_dir, subdir) for subdir in data]

    # Load the images
    train_pr.load_data(data)

    # preprocesses the data (in fact, just caches & prefetches)
    train_pr.preprocess()

    # trains the model
    history = train_pr.train_model()

    # saves the (re)trained model
    model_file_path = mlflow_tracking_dir + "/models/TL_TR_" + \
            create_run_timestamp().removeprefix("RUN") + "TS_" + \
            str(len(train_pr.class_names)) + 'cls_' + \
            str(train_pr.image_size[0]) + "px_" + \
            str(train_pr.batch_size) + "btc_" + \
            str(train_pr.fine_tune_epochs) + "fte_" + \
            "model.keras"
    
    mlflow.keras.save_model(train_pr, model_file_path)
    # train_pr.save_model(model_file_path)

    # Prediction (on the target [test] dataset)
    true_classes, predicted_classes = prdct(model_file_path, train_pr.test_ds)

    # prints Confusion Matrix & Classification Report
    show_clsf_rprt(true_classes, predicted_classes, train_pr.class_names)
    show_conf_mtrx(true_classes, predicted_classes)

    return train_pr


def re_training(train, hyper, data):
    train_pr = TrainPR(train)

    if hyper:
        train_pr.update_hyperparameters(**parse_hyper_list(hyper))
        mlflow.log_param(hyper)

    # Load the images
    train_pr.load_data(data)

    # preprocesses the data (in fact, just caches & prefetches)
    train_pr.preprocess()

    # trains the model
    history = train_pr.train_model()

    # saves the (re)trained model
    model_file_path = mlflow_tracking_dir + "/models/RL_TR_" + \
            create_run_timestamp().removeprefix("RUN") + "TS_" + \
            str(len(train_pr.class_names)) + 'cls_' + \
            str(train_pr.image_size[0]) + "px_" + \
            str(train_pr.batch_size) + "btc_" + \
            str(train_pr.fine_tune_epochs) + "fte_" + \
            "model.keras"
    
    mlflow.keras.save_model(train_pr, model_file_path)
    # train_pr.save_model(model_file_path)

    # Prediction (on the target [test] dataset)
    true_classes, predicted_classes = prdct(model_file_path, train_pr.test_ds)

    # prints Confusion Matrix & Classification Report
    show_clsf_rprt(true_classes, predicted_classes, train_pr.class_names)
    show_conf_mtrx(true_classes, predicted_classes)

    return train_pr


def print_xprmnt_info(experiment):
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")


def print_run_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in mlflow.MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


if __name__ == "__main__":
    print("Paramters check")
    args = cli_parameters()

    ## sets the default location for the 'mlruns' directory which represents the default local storage location for MLflow entities and artifacts
    # one of the ways to launch a web interface that displays run data stored in the 'mlruns' directory is the command line 'mlflow ui --backend-store-uri <MLFLOW_TRACK_DIR_PATH>'
    # Read the MLFLOW_TRACK_DIR_PATH environment variable
    mlflow_tracking_dir = os.getenv("MLFLOW_TRACK_DIR_PATH", "./MLFlow")
    model_images_dir = os.getenv("IMAGES_DIR_PATH", "./Data")
    mlflow.set_tracking_uri(mlflow_tracking_dir)

    ## logs metrics, parameters, and models without the need for explicit log statements
    # logs model signatures (describing model inputs and outputs), trained models (as MLflow model artifacts) & dataset information to the active fluent run
    mlflow.autolog()

    experiment_name = "PR_model"
    experiment_description = (
        "This is the plant recognition project. "
        "This experiment contains the images models for leafs."
    )
    experiment_tags = {
        "project_name": "plant-recognition",
        "team": "aal-ml",
        "mlflow.note.content": experiment_description,
    }
    experiment_id = create_mlflow_xprmnt(experiment_name=experiment_name, tags=experiment_tags)
    mlflow.set_experiment(experiment_name)

    run_name = create_run_timestamp()
    run_name = ("I" + run_name) if args.init else ("T" + run_name)
    run_tags = {"version": "v1", "priority": "P1"} # FIXME
    with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
        if args.init:
            first_training(args.hyper, args.data)
        else:
            re_training(args.train, args.hyper, args.data)
        
        # prints experiment data
        experiment = mlflow.get_experiment(experiment_id)
        print("experiment data:")
        print_xprmnt_info(experiment)

    print("********************** MLFlow_train end **********************")
    # Close the MLflow run
    mlflow.end_run()
