'''
script that performs the (re)training of the (target) model thus producing a new ([re]trained) model based on the output conditions of 'retrain_decider.py'
'''

import sys
sys.path.insert(1, '../')
from src.train_model import TrainPR
from src.predict_model import prdct, show_clsf_rprt, show_conf_mtrx
import mlflow
from mlflow_utils import create_mlflow_xprmnt, dt_stamp, print_run_info, print_xprmnt_info

# checking & fetching arguments (if any) of the command line 'python3 mlflow_train.py [-i] [-p <kwargs>] [-d <data_dir_paths>] [-m <model_file_path>]'
isInit = '-i' in sys.argv
iIdx = sys.argv.index("-i") if isInit else -1
pIdx = sys.argv.index("-p") if ('-p' in sys.argv) else -1
dIdx = sys.argv.index("-d") if ('-d' in sys.argv) else -1
mIdx = sys.argv.index("-m") if ('-m' in sys.argv) else -1
if isInit:
  assert iIdx == 1
  assert (dIdx + mIdx) == (-1 + -1)
  if pIdx > 0:
    assert len(sys.argv) > (pIdx + 1)
else:
  assert dIdx > 0
  assert mIdx > (dIdx + 1)
  assert len(sys.argv) > (mIdx + 1)
  if pIdx > 0:
    assert dIdx > (pIdx + 1)
kwargs_lst = [] # list of 'key=value' pairs for updating (hyper)parameters
keys_lst = []
values_lst = []
if pIdx > 0:
  kwargs_lst = [sys.argv[i] for i in range(len(sys.argv)) if (i > pIdx) if (i < dIdx)] if dIdx > 0 else \
    [sys.argv[i] for i in range(len(sys.argv)) if (i > pIdx)]
  assert all('=' in el for el in kwargs_lst)
  for el in kwargs_lst:
    el_splt_lst = el.split("=")
    keys_lst.append(el_splt_lst[0])
    values_lst.append(el_splt_lst[1])
  kwargs_dict = dict(zip(keys_lst, values_lst))
data_dir_path_lst = [] # list of directory paths to the target datasets
model_file_path_ = '' # file path to the target model used for retraining (redundant for the training mode i.e. command line flag '-i')
if dIdx > 0:
  data_dir_path_lst = [sys.argv[i] for i in range(len(sys.argv)) if (i > dIdx) if (i < mIdx)]
if mIdx > 0:
  model_file_path_ = sys.argv[mIdx + 1]

print("-p args:\n", kwargs_dict)
print("-d args:\n", data_dir_path_lst)
print("-m arg:\n", model_file_path_)

## Constants
BATCH_SIZE = 32
IMAGE_SIZE = (180, 180)
# argument for the command line option '-d <data_dir_paths>'
INIT_DATA_DIR_PATH = "/home/alex/Prj/WB/DST/MLO/data/subs/0" # FIXME initial value: "../../data"
MODEL_DIR_PATH = "/home/alex/Prj/WB/DST/MLO/mods/" # FIXME initial value: "../../models/"
BASE_LEARNING_RATE = 0.0001
FINE_TUNE_AT = 100
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
MLFLOW_TRACK_DIR_PATH = '/home/alex/Prj/WB/DST/MLO/mlflow/track' # FIXME

init_data_dir_path_ = []
init_data_dir_path_.append(INIT_DATA_DIR_PATH)

## sets the default location for the 'mlruns' directory which represents the default local storage location for MLflow entities and artifacts 
# one of the ways to launch a web interface that displays run data stored in the 'mlruns' directory is the command line 'mlflow ui --backend-store-uri <MLFLOW_TRACK_DIR_PATH>'
mlflow.set_tracking_uri(MLFLOW_TRACK_DIR_PATH)

## logs metrics, parameters, and models without the need for explicit log statements 
# logs model signatures (describing model inputs and outputs), trained models (as MLflow model artifacts) & dataset information to the active fluent run
mlflow.autolog()

# creates and sets (as active) experiments & assigns attributes
if pIdx == -1:
  experiment_name = "init_param_TL_models"
else:
  experiment_name = "modi_param_TL_models"
xprmnt_tags = {"env": "dev", "version": "1.0.0", "priority": 1} # FIXME
experiment_id = create_mlflow_xprmnt(experiment_name=experiment_name,
                          tags=xprmnt_tags)
mlflow.set_experiment(experiment_name)

# sets and starts a new mlflow run within the set experiment
if isInit:
  run_name = "trun_" + dt_stamp() + "TS"
else:
  run_name = "rtrun_" + dt_stamp() + "TS"
run_tags={"version": "v1", "priority": "P1"} # FIXME
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
    if pIdx > 0:
      train_pr.update_hyperparameters(**kwargs_dict)
    # Load data
    train_pr.load_data(data_dirs=init_data_dir_path_)
  else:
    train_pr = TrainPR(model_path=model_file_path_,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        base_learning_rate=BASE_LEARNING_RATE,
        fine_tune_at=FINE_TUNE_AT,
        initial_epochs=INITIAL_EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
                       )
    if pIdx > 0:
      train_pr.update_hyperparameters(**kwargs_dict)
    # Load data
    # NOTE: ideally, the information needs to be taken from the DB ('retrain_decider.py' module)
    train_pr.load_data(data_dirs=data_dir_path_lst)

  # preprocesses the data (in fact, just caches & prefetches)
  train_pr.preprocess()

  if isInit:
    # trains the model
    history = train_pr.train_model()
  else:
    # retrains the model
    history = train_pr.train_model(is_init=False)

  if isInit:
    mode = "t_"
  else:
    mode = "rt_"
  # saves the (re)trained model
  model_file_path = MODEL_DIR_PATH + "TL_" + mode + \
    dt_stamp() + "TS_" + \
      str(len(train_pr.class_names)) + 'cls_' + \
        str(train_pr.image_size[0]) + "px_" + \
          str(train_pr.batch_size) + "btc_" + \
            str(train_pr.fine_tune_epochs) + "fte_" + \
              "model.keras"
  train_pr.save_model(model_file_path)

  # Prediction (on the target [test] dataset)
  true_classes, predicted_classes = prdct(model_file_path, train_pr.test_ds)

  # prints Confusion Matrix & Classification Report
  show_clsf_rprt(true_classes, predicted_classes, train_pr.class_names)
  show_conf_mtrx(true_classes, predicted_classes)

# mlflow run finish line
print("---------------------mlflow_run=fin---------------------")

# prints experiment data
experiment = mlflow.get_experiment(experiment_id)
print("experiment data:")
print_xprmnt_info(experiment)

# prints run data
print("run data:")
print_run_info(mlflow.get_run(run_id=run.info.run_id))

# fin
print("**********************mlflow_train=fin**********************")
