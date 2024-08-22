'''
script that performs the (re)training of the (target) model thus producing a new ([re]trained) model based on the output conditions of 'retrain_decider.py'
'''

import sys
sys.path.insert(1, '../../')
from src.models.train_model import TrainPR
from src.models.predict_model import prdct, show_clsf_rprt, show_conf_mtrx
import mlflow
from mlflow_utils import create_mlflow_xprmnt, dt_stamp, print_run_info, print_xprmnt_info

# checking & fetching arguments (if any) of the command line 'python mlflow_train.py [-i] [-d <data_dir_paths>] [-t <model_file_path>]'
isInit = '-i' in sys.argv
iIdx = sys.argv.index("-i") if isInit else -1
dIdx = sys.argv.index("-d") if ('-d' in sys.argv) else -1
tIdx = sys.argv.index("-t") if ('-t' in sys.argv) else -1
if isInit:
  assert iIdx == 1
  if dIdx > 0:
    assert (len(sys.argv) > (dIdx + 1)) & (dIdx > iIdx)
    if tIdx > 0:
      assert (len(sys.argv) > (tIdx + 1)) & (tIdx > dIdx)
elif dIdx > 0:
  assert len(sys.argv) > (dIdx + 1)
  if tIdx > 0:
    assert (len(sys.argv) > dIdx) & (tIdx > dIdx)
data_dir_path_lst = [] # list of directory paths to the target datasets - FIXME: so far unused/idle
if dIdx > 0:
  data_dir_path_lst = [sys.argv[i] for i in range(len(sys.argv)) if (i > dIdx) if (i < tIdx)] if tIdx > 0 else \
    [sys.argv[i] for i in range(len(sys.argv)) if (i > dIdx)]
model_file_path_ = '' # file path to the target model used for retraining (redundant for the training mode i.e. command line flag '-i') - FIXME: so far unused/idle
if tIdx > 0:
  model_file_path_ = sys.argv[tIdx + 1]

## Constants
BATCH_SIZE = 32
IMAGE_SIZE = (180, 180)
# argument for the command line option '-d <data_dir_paths>'
INIT_DATA_DIR_PATH = "/home/alex/Prj/WB/DST/MLO/data/subs/0" # FIXME initial value: "../../data"
DATA_DIR_PATH = "/home/alex/Prj/WB/DST/MLO/data/subs/2" # FIXME initial value: "../../data"
# argument for the command line option '-t <model_file_path>'
INIT_MODEL_FILE_PATH = "../../models/TL_180px_32b_20e_model.keras" # FIXME
MODEL_DIR_PATH = "../../models/"
BASE_LEARNING_RATE = 0.0001
FINE_TUNE_AT = 100
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
SEED = 123
MLFLOW_TRACK_DIR_PATH = '/home/alex/Prj/WB/DST/MLO/mlflow/track' # FIXME default value: "http://tracking-server:5000"

## sets the default location for the 'mlruns' directory which represents the default local storage location for MLflow entities and artifacts 
# one of the ways to launch a web interface that displays run data stored in the 'mlruns' directory is the command line 'mlflow ui --backend-store-uri <MLFLOW_TRACK_DIR_PATH>'
mlflow.set_tracking_uri(MLFLOW_TRACK_DIR_PATH)

## logs metrics, parameters, and models without the need for explicit log statements 
# logs model signatures (describing model inputs and outputs), trained models (as MLflow model artifacts) & dataset information to the active fluent run
mlflow.autolog()

# creates and sets (as active) experiments & assigns attributes
experiment_name = "init_param_TL_models" # FIXME
xprmnt_tags = {"env": "dev", "version": "1.0.0", "priority": 1} # FIXME
experiment_id = create_mlflow_xprmnt(experiment_name=experiment_name,
                          tags=xprmnt_tags)
mlflow.set_experiment(experiment_name)

# sets and starts a new mlflow run within the set experiment
if isInit:
  run_name = "trun_" + dt_stamp() + "TS"
else:
  run_name = "rtrun_" + dt_stamp() + "TS"
run_tags={"version": "v1", "priority": "P1"}
with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
  if isInit:
    # Create an instance of TrainPR
    train_pr = TrainPR(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        base_learning_rate=BASE_LEARNING_RATE,
        fine_tune_at=FINE_TUNE_AT,
        initial_epochs=INITIAL_EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
    )
    # Load data
    train_pr.load_data(data_dir=INIT_DATA_DIR_PATH, seed=SEED)
    # FIXME: for 'isInit' case, since 'model' object is 'None', no attributes are given and thus cannot receive assigned values
    # FIXME: generally, 'model' attributes need to be reconsidered & remastered
  else:
    train_pr = TrainPR(model_path=INIT_MODEL_FILE_PATH)
    # Load data
    # TODO: The information has to be taken from the DB ('retrain_decider.py' module)
    train_pr.load_data(data_dir=DATA_DIR_PATH)  
    # FIXME: since 'load_data()' populates 'class_names' from 'train_ds' of 'data_dir', the new (added) dataset alone cannot be used without its code adjustment

  # preprocesses the data (in fact, just caches & prefetches)
  train_pr.preprocess()

  # builds the model (essentially, stacks up the layers)
  train_pr.build_model()
  # FIXME: since 'build_data()' uses the size of the populated-via-'load_data()' 'class_names' attribute for shaping of the 'prediction_layer', its code needs adjustment

  if isInit:
    # trains the model
    history = train_pr.train_model()
    # FIXME: 'base_model.trainable = True' has to be put before the for loop that freezes layers up to 'FINE_TUNE_AT'
    # FIXME: 'initial_epoch=len(history.epoch)' is vulnerable for the 'not isInit' case since in this case 'history' is not-yet-defined
    # FIXME: better to avoid 'initial_epoch' for 'not isInit' cases
  else:
    # retrains the model
    history = train_pr.train_model(is_init=False)

  # saves the (re)trained model
  # FIXME: fetch the metadata from within the train_pr object
  model_file_path = MODEL_DIR_PATH + "TL_" + \
    dt_stamp() + "TS_" + \
      train_pr.image_size[0] + "px_" + \
        train_pr.batch_size + "b_" + \
          train_pr.fine_tune_epochs + "fte_" + \
            "model.keras"
  train_pr.save_model(model_file_path)

  # Prediction (on the target [test] dataset)
  true_classes, predicted_classes = prdct(model_file_path, train_pr.test_ds)

  # prints Confusion Matrix & Classification Report
  print("confusion matrix:")
  show_clsf_rprt(model_file_path, train_pr.test_ds)
  print("classification report:")
  show_conf_mtrx(model_file_path, train_pr.test_ds)

# prints experiment data
experiment = mlflow.get_experiment(experiment_id)
print_xprmnt_info(experiment)

# prints run data
print_run_info(mlflow.get_run(run_id=run.info.run_id))
