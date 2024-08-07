import pytest
from unittest.mock import MagicMock, patch

# Mock TensorFlow imports to avoid loading TensorFlow during testing
with patch.dict('sys.modules', {'tensorflow': MagicMock(), 'tensorflow.keras': MagicMock(), 'tensorflow.keras.layers': MagicMock(), 'tensorflow.keras.applications': MagicMock()}):
    from models.train_model import TrainPR

# Constants for testing
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
BASE_LEARNING_RATE = 0.0001
FINE_TUNE_AT = 100
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

@pytest.fixture
def train_pr():
    return TrainPR(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        base_learning_rate=BASE_LEARNING_RATE,
        fine_tune_at=FINE_TUNE_AT,
        initial_epochs=INITIAL_EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS
    )

@patch("models.train_model.tf.keras.utils.image_dataset_from_directory")
def test_load_data(mock_image_dataset, train_pr):
    # Arrange
    mock_image_dataset.return_value = MagicMock()

    # Act
    train_pr.load_data("fake_data_dir")

    # Assert
    mock_image_dataset.assert_called()
    assert train_pr.train_ds is not None
    assert train_pr.val_ds is not None
    assert train_pr.test_ds is not None

@patch("models.train_model.tf.keras.Model.compile")
@patch("models.train_model.tf.keras.Input")
@patch("models.train_model.MobileNetV2")
def test_build_model(mock_mobilenet, mock_input, mock_compile, train_pr):
    # Arrange
    mock_input.return_value = MagicMock()
    mock_mobilenet.return_value = MagicMock()

    # Act
    train_pr.build_model(num_classes=10)

    # Assert
    mock_mobilenet.assert_called_once()
    mock_compile.assert_called_once()
    assert train_pr.model is not None

@patch("models.train_model.tf.keras.Model.fit")
def test_train_model(mock_fit, train_pr):
    # Arrange
    train_pr.model = MagicMock()

    # Act
    history = train_pr.train_model()

    # Assert
    train_pr.model.fit.assert_called()
    assert history is not None

@patch("models.train_model.tf.keras.Model.save")
def test_save_model(mock_save, train_pr):
    # Arrange
    train_pr.model = MagicMock()

    # Act
    train_pr.save_model("fake_model_path")

    # Assert
    train_pr.model.save.assert_called_once_with("fake_model_path")
