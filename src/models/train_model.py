# import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


def load_data(data_dir: str, image_size: tuple, batch_size: int) -> tuple:
    """
    Loads and prepares the training and validation datasets from a directory.

    Args:
        data_dir (str): The directory containing the image dataset.
        image_size (tuple): The desired size of the images in the dataset.
        batch_size (int): The number of samples per batch.

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
    )
    return train_ds, val_ds


def preprocess(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ratio: int) -> tuple:
    """
    Preprocesses the training, validation, and test datasets.

    Args:
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
        test_ratio (int): The ratio of validation batches to be used for testing.

    Returns:
        tuple: A tuple containing the preprocessed training, validation, and test datasets.
    """
    val_batches = val_ds.cardinality()
    test_ds = val_ds.take(val_batches // test_ratio) # 4% of entire dataset
    val_ds = val_ds.skip(val_batches // test_ratio) # 16% of entire dataset
    print("Number of validation batches: %d" % val_ds.cardinality())
    print("Number of test batches: %d" % test_ds.cardinality())

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_model(input_shape: tuple, num_classes: int, base_learning_rate: float) -> tf.keras.Model:
    """
    Builds a model using MobileNetV2 as the base model.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of classes for classification.
        base_learning_rate (float): The learning rate for the optimizer.

    Returns:
        tf.keras.Model: The built model.

    """
    base_model = MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense(num_classes, activation="softmax")

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def train_model(model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, initial_epochs: int, callbacks: list) -> tf.keras.callbacks.History:
    """
    Trains a machine learning model using the given training and validation datasets.

    Args:
        model (tf.keras.Model): The machine learning model to train.
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
        initial_epochs (int): The number of initial epochs to train the model.
        callbacks (list): List of callbacks to be used during training.

    Returns:
        tf.keras.callbacks.History: The training history of the model.
    """
    history = model.fit(
        train_ds,
        epochs=initial_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    return history


def fine_tune_model(
    model: tf.keras.Model, base_model: tf.keras.Model, fine_tune_at: int, base_learning_rate: float, fine_tune_epochs: int, callbacks: list
) -> tf.keras.callbacks.History:
    """
    Fine-tunes a model by freezing the layers of a base model up to a specified index,
    compiling the model with a specified learning rate, loss function, and metrics,
    and training the model on a training dataset for a specified number of epochs.

    Args:
        model (tf.keras.Model): The model to be fine-tuned.
        base_model (tf.keras.Model): The base model whose layers will be frozen.
        fine_tune_at (int): The index of the layer up to which the base model layers will be frozen.
        base_learning_rate (float): The learning rate for the optimizer.
        fine_tune_epochs (int): The number of epochs to train the model.
        callbacks (list): List of callbacks to be used during training.

    Returns:
        tf.keras.History: The training history of the fine-tuned model.
    """
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_ds,
        epochs=fine_tune_epochs,
        initial_epoch=len(history.epoch),
        validation_data=val_ds,
        callbacks=callbacks,
    )

    return history_fine


def save_model(model: tf.keras.Model, filename: str) -> None:
    """
    Saves the given model to a file.

    Args:
        model: The model to be saved.
        filename: The name of the file to save the model to.

    Returns:
        None
    """
    model.save(filename)


# The condition __name__ == "__main__" is used in a Python program to execute the code inside the if statement only
# when the program is executed directly by the Python interpreter.
# When the code in the file is imported as a module the code inside the if statement is not executed.
if __name__ == "__main__":
    # Constants
    BATCH_SIZE = 32
    IMAGE_SIZE = (180, 180)
    DATA_DIR = "data/raw"
    TEST_RATIO = 5
    BASE_LEARNING_RATE = 0.0001
    FINE_TUNE_AT = 100
    INITIAL_EPOCHS = 10
    FINE_TUNE_EPOCHS = 10

    # This line was commented cause generated a model with overfitting.
    # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Load data
    train_ds, val_ds = load_data(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    NUM_CLASSES = len(train_ds.class_names)

    # Preprocess data
    train_ds, val_ds, test_ds = preprocess(train_ds, val_ds, TEST_RATIO)

    # Build model
    model = build_model(IMAGE_SIZE + (3,), NUM_CLASSES, BASE_LEARNING_RATE)
    print(type(model))
    print(model.summary())

    # Train model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, verbose=1, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.2, patience=3, verbose=1, cooldown=5
        ),
    ]
    history = train_model(model, train_ds, val_ds, INITIAL_EPOCHS, callbacks)

    # Fine-tune model
    base_model = model.layers[1]
    history_fine = fine_tune_model(
        model,
        base_model,
        FINE_TUNE_AT,
        BASE_LEARNING_RATE,
        INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        callbacks,
    )

    # Save model
    save_model(model, DATA_DIR.replace("raw", "") + "TL_180px_32b_20e_model.keras")
