# here comes the script that performs the retraining of the (target) model thus producing a new (retrained) model based on the output conditions of 'retrain_decider.py'

# python mlflow_train.py -d/--data <data_path> [-i/--init] -t/--train <model_path>

from train_model import TrainPR
import mlflow
from sklearn.metrics import classification_report, confusion_matrix

# Use the MLFlow logging and setting of the environment for the execution of the experiment
# TODO: put the code here

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (180, 180)
# argument for the script: -d/--data <data_path>
DATA_PATH = "../../data"
# argument for the script: -t/--train <model_path>
MODEL_FILENAME = "../../data"
BASE_LEARNING_RATE = 0.0001
FINE_TUNE_AT = 100
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

with mlflow.start_run():
    if ("-i"):
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
      train_pr.load_data(DATA_PATH) # The folder has to be 0
    else:
      # argument for the script: -t/--train and -d/--data
      train_pr = TrainPR(MODEL_FILENAME)

      # Load data
      # The information has to be taken from the DB (retrain_decider.py module)
      train_pr.load_data(DATA_PATH) # we need to merge the folders starting from

    # Preprocess data
    train_pr.preprocess()

    # Build model
    train_pr.build_model()

    if ("-i"):
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
