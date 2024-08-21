import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import numpy as np


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
            val_ds (Optional[tf.data.Dataset]): The validation dataset, initialized as None.
            test_ds (Optional[tf.data.Dataset]): The test dataset, initialized as None.
        """
        self.model = None
        if model_path:
            # print("Loading model from path:", model_path)  # Debug statement
            self.load_model(model_path)
            # print("Model loaded:", self.model)  # Debug statement
        else:
            self.image_size = kwargs.get("image_size")
            self.batch_size = kwargs.get("batch_size")
            self.base_learning_rate = kwargs.get("base_learning_rate")
            self.fine_tune_at = kwargs.get("fine_tune_at")
            self.initial_epochs = kwargs.get("initial_epochs")
            self.fine_tune_epochs = kwargs.get("fine_tune_epochs")
            # self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def load_model(self, model_path: str):
        """
        Load a model from the specified path and restore hyperparameters.

        Args:
            model_path (str): The path to the saved model file.
        """
        # See https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
        self.model = tf.keras.models.load_model(model_path)
        # print("Inside load_model, self.model =", self.model)  # Debug statement
        if self.model is not None:
            self.image_size = tuple(self.model.image_size)
            self.batch_size = self.model.batch_size
            self.base_learning_rate = self.model.base_learning_rate
            self.fine_tune_at = self.model.fine_tune_at
            self.initial_epochs = self.model.initial_epochs
            self.fine_tune_epochs = self.model.fine_tune_epochs

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

    def load_data(self, data_dir: str, seed: int):
        """
        Loads the training, validation, and test datasets from the specified directory.

        Args:
            data_dir (str): The directory where the data is stored. It should contain
                            'training' and 'test' subdirectories.
            seed (int): Used to make the behavior of the initializer deterministic.

        Sets:
            self.train_ds: The training dataset, which is 80% of the data in the 'training' directory.
            self.val_ds: The validation dataset, which is 20% of the data in the 'training' directory.
            self.test_ds: The test dataset, which is all the data in the 'test' directory.
        """
        # Create a validation set from the training set (20% of training data)
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )

        self.model.class_names = self.train_ds.class_names
        val_batches = tf.data.experimental.cardinality(self.val_ds)
        self.test_ds = self.val_ds.take(val_batches // 2) # 10% of dataset
        self.val_ds = self.val_ds.skip(val_batches // 2)  # 10% of dataset

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
        num_classes = len(self.model.class_names)
        base_model = MobileNetV2(
            input_shape=self.image_size + (3,), include_top=False, weights="imagenet"
        )
        global_average_layer = layers.GlobalAveragePooling2D()
        prediction_layer = layers.Dense(num_classes, activation="softmax")

        inputs = tf.keras.Input(shape=self.image_size + (3,))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = layers.Dropout(0.2)(x)
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
                patience=10,
                verbose=1,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy", factor=0.2, patience=3, verbose=1, cooldown=5
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

            # Fine-tune model
            for layer in self.model.layers[: self.fine_tune_at]:
                layer.trainable = False
            
            epochs = self.initial_epochs + self.fine_tune_epochs
            init_epoch = len(history.epoch)
        else:
            history = None  # No history from initial training
            epochs = self.fine_tune_epochs
            init_epoch = 0

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
        # Save the hyperparameters in the model object
        self.model.image_size = self.image_size
        self.model.batch_size = self.batch_size
        self.model.base_learning_rate = self.base_learning_rate
        self.model.fine_tune_at = self.fine_tune_at
        self.model.initial_epochs = self.initial_epochs
        self.model.fine_tune_epochs = self.fine_tune_epochs

        # Save the model
        self.model.save(filename)


# The condition __name__ == "__main__" is used in a Python program to execute the code inside the if statement only
# when the program is executed directly by the Python interpreter.
# When the code in the file is imported as a module the code inside the if statement is not executed.
if __name__ == "__main__":
    # Constants
    BATCH_SIZE = 32
    IMAGE_SIZE = (180, 180)
    DATA_DIR = "data"
    BASE_LEARNING_RATE = 0.0001
    FINE_TUNE_AT = 100
    INITIAL_EPOCHS = 10
    FINE_TUNE_EPOCHS = 10

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
    train_pr.load_data()

    # Preprocess data
    train_pr.preprocess()

    # Build model
    train_pr.build_model()
    print(type(train_pr.model))
    print(train_pr.model.summary())

    # Train model
    history, history_fine = train_pr.train_model()

    # Save model
    train_pr.save_model("models/" + "TL_180px_32b_20e_model.keras")
