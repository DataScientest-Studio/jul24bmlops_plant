# here comes the script that performs the retraining of the (target) model thus producing a new (retrained) model based on the output conditions of 'retrain_decider.py'

# python mlflow_train.py -d/--data <data_path> [-i/--init] -t/--train <model_path>

import argparse
import sys

import mlflow
from sklearn.metrics import classification_report, confusion_matrix
from train_model import TrainPR

# Use the MLFlow logging and setting of the environment for the execution of the experiment
# TODO: put the code here


def main():
    parser = argparse.ArgumentParser(
        description="Train a Plant Recognition ML model with MLflow"
    )

    # Optional arguments
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Path to the data (required for both initialization and training)",
    )
    parser.add_argument(
        "-i",
        "--init",
        action="store_true",
        help="Initialize the model with the provided data",
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        help="Path to save the trained model (requires data path)",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.init and args.train:
        print("Error: -i/--init and -t/--train cannot be used together.")
        sys.exit(1)

    if args.init and not args.data:
        print("Error: -i/--init requires a data path (-d/--data).")
        sys.exit(1)

    if args.train and not args.data:
        print("Error: -t/--train requires a data path (-d/--data).")
        sys.exit(1)

    if not args.init and not args.train:
        print("Error: You must specify either -i/--init or -t/--train.")
        sys.exit(1)

    # Constants
    BATCH_SIZE = 32
    IMAGE_SIZE = (180, 180)
    # argument for the script: -d/--data <data_path>
    DATA_PATH = args.data # "../../data"
    # argument for the script: -t/--train <model_path>
    MODEL_FILENAME = args.train #"../../data"
    BASE_LEARNING_RATE = 0.0001
    FINE_TUNE_AT = 100
    INITIAL_EPOCHS = 10
    FINE_TUNE_EPOCHS = 10

    with mlflow.start_run():
        if args.init:
            # Create an instance of TrainPR
            # argument for the script: -i/--init
            train_pr = TrainPR(
                image_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                base_learning_rate=BASE_LEARNING_RATE,
                fine_tune_at=FINE_TUNE_AT,
                initial_epochs=INITIAL_EPOCHS,
                fine_tune_epochs=FINE_TUNE_EPOCHS,
            )
            # Load data
            train_pr.load_data(DATA_PATH)  # The folder has to be 0
        else:
            # argument for the script: -t/--train and -d/--data
            train_pr = TrainPR(MODEL_FILENAME)

            # Load data
            # The information has to be taken from the DB (retrain_decider.py module)
            train_pr.load_data(DATA_PATH)  # we need to merge the folders starting from

        # Preprocess data
        train_pr.preprocess()

        # Build model
        train_pr.build_model()

        if args.init:
            # Training model
            history = train_pr.train_model()
        else:
            # Retraining model
            history = train_pr.train_model(is_init=False)

        # Save model
        # Define the logic that will keep track of the models: we can use date, tags, etc.
        train_pr.save_model("../../models/" + "TL_180px_32b_20e_model.keras")

        # Prediction
        predicted_classes, test_classes = train_pr.predict()

        # Confusion Matrix
        cm = confusion_matrix(test_classes, predicted_classes)
        print(classification_report(test_classes, predicted_classes))


    if __name__ == "__main__":
        main()
