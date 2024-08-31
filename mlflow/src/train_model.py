import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix


# hyperparameters:
# Learning Rate: Controls the step size during gradient descent.
# Batch Size: Number of samples processed before the model’s internal parameters are updated.
# Number of Epochs: Number of complete passes through the training dataset.
# Dropout Rate: Fraction of neurons randomly dropped during training to prevent overfitting.
# Optimizer Type: Algorithm used to update model parameters (e.g., Adam, SGD).
# Number of Layers and Units: Architecture of the neural network, including the number of layers and neurons per layer.
class TrainPR:
    def __init__(
            self,
            model_path=None,
            **kwargs
    ):
        """
        Initialize the training configuration.

        Args:
            model_path (str, optional): The path to a previously saved model. If provided, the model will be loaded from this path.
            kwargs: Other hyperparameters, which are saved in self.model when provided.

            image_size (tuple): The size of the input images (height, width).
            batch_size (int): The number of samples per batch.
            base_learning_rate (float): The initial learning rate for training.
            fine_tune_at (int): The layer at which to start fine-tuning.
            initial_epochs (int): The number of epochs to train before fine-tuning.
            fine_tune_epochs (int): The number of epochs to fine-tune the model.

        Attributes:
            image_size (tuple): The size of the input images (height, width).
            batch_size (int): The number of samples per batch.
            base_learning_rate (float): The initial learning rate for training.
            fine_tune_at (int): The layer at which to start fine-tuning.
            initial_epochs (int): The number of epochs to train before fine-tuning.
            fine_tune_epochs (int): The number of epochs to fine-tune the model.
            model (Optional[tf.keras.Model]): The model to be trained, initialized as None.
            train_ds (Optional[tf.data.Dataset]): The training dataset, initialized as None.
            class_names (Optional[list[str]]): The list of class names from the training dataset, initialized as None.
            val_ds (Optional[tf.data.Dataset]): The validation dataset, initialized as None.
            test_ds (Optional[tf.data.Dataset]): The test dataset, initialized as None.
        """
        self.train_ds = None
        self.class_names = None
        self.val_ds = None
        self.test_ds = None
        self.model = None
        self.seed = 123
        self.validation_split = 0.2
        self.val_tst_split = 2
        self.num_classes = 38
        self.chnls = (3,)
        self.dropout_rate = 0.2
        self.init_weights = "imagenet"
        self.patience_EarlyStopping = 10
        self.patience_ReduceLROnPlateau = 3
        self.factor_ReduceLROnPlateau = 0.2
        self.cooldown_ReduceLROnPlateau = 5
        if model_path:
            # print("Loading model from path:", model_path)  # Debug statement
            self.load_model(model_path)
        self.image_size = kwargs.get("image_size")
        self.batch_size = kwargs.get("batch_size")
        self.base_learning_rate = kwargs.get("base_learning_rate")
        self.fine_tune_at = kwargs.get("fine_tune_at")
        self.initial_epochs = kwargs.get("initial_epochs")
        self.fine_tune_epochs = kwargs.get("fine_tune_epochs")

    def load_model(self, model_path: str):
        """
        Load a model from the specified path and restore hyperparameters.

        Args:
            model_path (str): The path to the saved model file.
        """
        # See https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
        self.model = tf.keras.models.load_model(model_path)
        # print("Inside load_model, self.model =", self.model)  # Debug statement

    def update_hyperparameters(self, **kwargs):
        """
        Update hyperparameters of the model.

        Args:
            kwargs: Hyperparameters to update. Possible keys are:
                    - image_size (tuple)
                    - batch_size (int)
                    - base_learning_rate (float)
                    - fine_tune_at (int)
                    - initial_epochs (int)
                    - fine_tune_epochs (int)
        """
        self.image_size = kwargs.get("image_size", self.image_size)
        self.batch_size = kwargs.get("batch_size", self.batch_size)
        self.base_learning_rate = kwargs.get("base_learning_rate", self.base_learning_rate)
        self.fine_tune_at = kwargs.get("fine_tune_at", self.fine_tune_at)
        self.initial_epochs = kwargs.get("initial_epochs", self.initial_epochs)
        self.fine_tune_epochs = kwargs.get("fine_tune_epochs", self.fine_tune_epochs)

    def load_data(self, data_dirs: list[str]):
        """
        Loads the training, validation, and test datasets from the specified directory.

        Args:
            data_dir (list of str): A list of directories where the data is stored. Each directory should contain
                                 'training' and 'test' subdirectories.

        Sets:
            self.train_ds: The training dataset, which is 80% of the data in the 'training' directory.
            self.val_ds: The validation dataset, which is 20% of the data in the 'training' directory.
            self.test_ds: The test dataset, which is all the data in the 'test' directory.
        """
        # Initialize lists to collect datasets from multiple directories
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for data_dir in data_dirs:
            # Load training and validation datasets from the current directory
            # Create a validation set from the training set (20% of training data)
            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=self.validation_split,
                subset="training",
                seed=self.seed,
                image_size=self.image_size,
                batch_size=self.batch_size,
            )
            if len(train_ds.class_names) == self.num_classes:
                self.class_names = train_ds.class_names

            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=self.validation_split,
                subset="validation",
                seed=self.seed,
                image_size=self.image_size,
                batch_size=self.batch_size,
            )

            val_batches = tf.data.experimental.cardinality(val_ds)
            test_ds = val_ds.take(val_batches // self.val_tst_split) # 10% of dataset
            val_ds = val_ds.skip(val_batches // self.val_tst_split)  # 10% of dataset
            # Add the loaded datasets to the respective lists
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)

        assert len(self.class_names) == self.num_classes

        # Concatenate all datasets from the different directories
        self.train_ds = train_datasets[0]
        for ds in train_datasets[1:]:
            self.train_ds = self.train_ds.concatenate(ds)

        self.val_ds = val_datasets[0]
        for ds in val_datasets[1:]:
            self.val_ds = self.val_ds.concatenate(ds)

        self.test_ds = test_datasets[0]
        for ds in test_datasets[1:]:
            self.test_ds = self.test_ds.concatenate(ds)      

    def preprocess(self):
        """
        Preprocess the datasets by caching and prefetching.

        This method applies caching and prefetching to the training, validation,
        and test datasets to improve performance during model training and evaluation.
        Caching stores the datasets in memory after the first epoch, and prefetching
        overlaps the data preprocessing and model execution to reduce the data loading
        bottleneck.

        Attributes:
            train_ds (tf.data.Dataset): The training dataset.
            val_ds (tf.data.Dataset): The validation dataset.
            test_ds (tf.data.Dataset): The test dataset.
        """
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def build_model(self):
        """
        Builds and compiles a MobileNetV2-based model for image classification.

        Returns:
            None
        """
        num_classes = len(self.class_names)
        base_model = MobileNetV2(
            input_shape=self.image_size + self.chnls, include_top=False, weights=self.init_weights
        )
        # Freeze the base model
        base_model.trainable = False
        global_average_layer = layers.GlobalAveragePooling2D()
        prediction_layer = layers.Dense(num_classes, activation="softmax")

        inputs = tf.keras.Input(shape=self.image_size + self.chnls)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = prediction_layer(x)

        self.model = tf.keras.Model(inputs, outputs)

    def train_model(self, is_init: bool=True):
        """
        Trains the model using the training dataset and validates it using the validation dataset.

        This method sets up two callbacks:
        1. EarlyStopping: Stops training when the validation accuracy has stopped improving for 10 epochs.
        - monitor: Metric to be monitored ('val_accuracy').
        - patience: Number of epochs with no improvement after which training will be stopped.
        - verbose: Verbosity mode.
        - restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored quantity.

        2. ReduceLROnPlateau: Reduces the learning rate when the validation accuracy has stopped improving.
        - monitor: Metric to be monitored ('val_accuracy').
        - factor: Factor by which the learning rate will be reduced.
        - patience: Number of epochs with no improvement after which learning rate will be reduced.
        - verbose: Verbosity mode.
        - cooldown: Number of epochs to wait before resuming normal operation after learning rate has been reduced.

        The model is trained for a specified number of initial epochs, and the training history is returned.

        Args:
            is_init (bool): To distinguish between training and retraining mode.

        Returns:
            history: A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=self.patience_EarlyStopping,
                verbose=1,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy", 
                factor=self.factor_ReduceLROnPlateau, 
                patience=self.patience_ReduceLROnPlateau, 
                verbose=1, 
                cooldown=self.cooldown_ReduceLROnPlateau
            ),
        ]

        if is_init:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

            history = self.model.fit(
                self.train_ds,
                epochs=self.initial_epochs,
                validation_data=self.val_ds,
                callbacks=callbacks,
            )
            
            epochs = self.initial_epochs + self.fine_tune_epochs
            init_epoch = len(history.epoch)
        else:
            history = None  # No history from initial training
            epochs = self.fine_tune_epochs
            init_epoch = 0

        self.model.trainable = True
        # Fine-tune model
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.model.layers[: self.fine_tune_at]:
            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        history = self.model.fit(
            self.train_ds,
            epochs=epochs,
            initial_epoch=init_epoch,
            validation_data=self.val_ds,
            callbacks=callbacks,
        )

        return history

    def predict(self):
        """
        Predicts classes for test dataset using the trained model.

        Returns:
            predicted_classes (numpy.array): Array of predicted classes.
            test_classes (numpy.array): Array of true classes.
        """
        test_classes = np.array([])
        predicted_classes = np.array([])

        for x, y in self.test_ds:
            predicted_classes = np.concatenate([predicted_classes, np.argmax(self.model(x, training=False), axis=-1)]).astype(int)
            test_classes = np.concatenate([test_classes, y.numpy()]).astype(int)
        
        return predicted_classes, test_classes

    def save_model(self, filename):
        """
        Save the current model to a file.

        Parameters:
        filename (str): The path to the file where the model will be saved.
        """
        # Save the model
        self.model.save(filename)


# The condition __name__ == "__main__" is used in a Python program to execute the code inside the if statement only
# when the program is executed directly by the Python interpreter.
# When the code in the file is imported as a module the code inside the if statement is not executed.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or initialize a machine learning model with MLflow"
    )

    # Optional arguments
    parser.add_argument(
        "-d",
        "--data",
        type=str,
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
    # argument for the script: -d/--data <data_path_list>
    DATA_PATH_LIST = args.data # "../../data"
    # argument for the script: -t/--train <model_path>
    MODEL_FILENAME = args.train #"../../data"
    BASE_LEARNING_RATE = 0.0001
    FINE_TUNE_AT = 100
    INITIAL_EPOCHS = 10
    FINE_TUNE_EPOCHS = 10
    SEED = 123

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
        train_pr.load_data(DATA_PATH_LIST, SEED)  # The folder has to be 0
    else:
        # argument for the script: -t/--train and -d/--data
        train_pr = TrainPR(MODEL_FILENAME)

        # Load data
        # The information has to be taken from the DB (retrain_decider.py module)
        train_pr.load_data(DATA_PATH_LIST, SEED)  # we need to merge the folders starting from

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
