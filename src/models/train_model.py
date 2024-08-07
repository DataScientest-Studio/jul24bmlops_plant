import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


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
        image_size: tuple,
        batch_size: int,
        base_learning_rate: float,
        fine_tune_at: int,
        initial_epochs: int,
        fine_tune_epochs: int,
    ):
        """
        Initialize the training configuration.

        Args:
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
        self.image_size = image_size
        self.batch_size = batch_size
        self.base_learning_rate = base_learning_rate
        self.fine_tune_at = fine_tune_at
        self.initial_epochs = initial_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def load_data(self, data_dir: str):
        """
        Loads the training, validation, and test datasets from the specified directory.

        Args:
            data_dir (str): The directory where the data is stored. It should contain
                            'training' and 'test' subdirectories.

        Sets:
            self.train_ds: The training dataset, which is 80% of the data in the 'training' directory.
            self.val_ds: The validation dataset, which is 20% of the data in the 'training' directory.
            self.test_ds: The test dataset, which is all the data in the 'test' directory.
        """
        train_dir = os.path.join(data_dir, "training")
        test_dir = os.path.join(data_dir, "test")

        # Create a validation set from the training set (20% of training data)
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )

        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )

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

    def build_model(self, num_classes: int):
        """
        Builds and compiles a MobileNetV2-based model for image classification.

        Args:
            num_classes (int): The number of output classes for the classification task.

        Returns:
            None
        """
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
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def train_model(self):
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

        Returns:
            history: A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.
            history_fine: A History object for the fine-tuning phase. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.
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

        history = self.model.fit(
            self.train_ds,
            epochs=self.initial_epochs,
            validation_data=self.val_ds,
            callbacks=callbacks,
        )

        # Fine-tune model
        base_model = self.model.layers[1]

        for layer in base_model.layers[: self.fine_tune_at]:
            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        history_fine = self.model.fit(
            self.train_ds,
            epochs=self.initial_epochs + self.fine_tune_epochs,
            initial_epoch=len(history.epoch),
            validation_data=self.val_ds,
            callbacks=callbacks,
        )

        return history, history_fine

    def save_model(self, filename):
        """
        Save the current model to a file.

        Parameters:
        filename (str): The path to the file where the model will be saved.
        """
        self.model.save(filename)


# The condition __name__ == "__main__" is used in a Python program to execute the code inside the if statement only
# when the program is executed directly by the Python interpreter.
# When the code in the file is imported as a module the code inside the if statement is not executed.
if __name__ == "__main__":
    # Constants
    BATCH_SIZE = 32
    IMAGE_SIZE = (180, 180)
    DATA_DIR = "data/raw"
    BASE_LEARNING_RATE = 0.0001
    FINE_TUNE_AT = 100
    INITIAL_EPOCHS = 10
    FINE_TUNE_EPOCHS = 10

    # Create an instance of TrainPR
    train_pr = TrainPR(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        base_learning_rate=BASE_LEARNING_RATE,
        fine_tune_at=FINE_TUNE_AT,
        initial_epochs=INITIAL_EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
    )

    # Load data
    train_pr.load_data()
    NUM_CLASSES = len(train_pr.train_ds.class_names)

    # Preprocess data
    train_pr.preprocess()

    # Build model
    train_pr.build_model(num_classes=NUM_CLASSES)
    print(type(train_pr.model))
    print(train_pr.model.summary())

    # Train model
    history, history_fine = train_pr.train_model()

    # Save model
    train_pr.save_model(DATA_DIR.replace("raw", "") + "TL_180px_32b_20e_model.keras")
