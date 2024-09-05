import pytest
from unittest.mock import MagicMock, patch
from model.train_model import TrainPR

@pytest.fixture
def mock_train_pr():
    with patch('model.train_model.tf.keras.models.load_model') as mock_load_model:
        mock_load_model.return_value = MagicMock()  # Mock the model
        train_pr = TrainPR(model_path="mock_model_path", image_size=(224, 224), batch_size=32)
        train_pr.load_model = MagicMock()  # Mock the `load_model` method
        return train_pr

def test_init_with_model_path():
    # Initialize without model_path to ensure model is None initially
    train_pr = TrainPR()
    assert train_pr.model is None  # Model should be None at initialization

    # Now test the case where a model_path is provided and the model is loaded
    with patch('train_model.tf.keras.models.load_model') as mock_load_model:
        mock_load_model.return_value = MagicMock()  # Mock the loaded model
        train_pr_with_model = TrainPR(model_path="mock_model_path")
        mock_load_model.assert_called_once_with("mock_model_path")
        assert train_pr_with_model.model is not None  # Model should no longer be None after loading

def test_load_model():
    with patch('model.train_model.tf.keras.models.load_model') as mock_load_model:
        mock_load_model.return_value = MagicMock()
        train_pr = TrainPR(model_path="mock_model_path")
        # Now `load_model` should have been called once during initialization
        mock_load_model.assert_called_once_with("mock_model_path")

def test_hyperparameters_initialization():
    # Initialize TrainPR with custom hyperparameters
    train_pr = TrainPR(image_size=(200, 200), batch_size=64, base_learning_rate=0.01)
    assert train_pr.image_size == (200, 200)
    assert train_pr.batch_size == 64
    assert train_pr.base_learning_rate == 0.01

def test_load_data(mock_train_pr):
    mock_train_pr.load_data = MagicMock()
    mock_train_pr.load_data(["mock_data_dir"], seed=42)
    mock_train_pr.load_data.assert_called_once_with(["mock_data_dir"], seed=42)

def test_preprocess(mock_train_pr):
    mock_train_pr.preprocess = MagicMock()
    mock_train_pr.preprocess()
    mock_train_pr.preprocess.assert_called_once()

def test_build_model(mock_train_pr):
    mock_train_pr.build_model = MagicMock()
    mock_train_pr.build_model()
    mock_train_pr.build_model.assert_called_once()

def test_train_model(mock_train_pr):
    mock_train_pr.train_model = MagicMock()
    history = mock_train_pr.train_model(is_init=True)
    mock_train_pr.train_model.assert_called_once_with(is_init=True)

def test_predict(mock_train_pr):
    mock_train_pr.predict = MagicMock(return_value=(["mock_prediction"], ["mock_test_class"]))
    predicted_classes, test_classes = mock_train_pr.predict()
    mock_train_pr.predict.assert_called_once()
    assert predicted_classes == ["mock_prediction"]
    assert test_classes == ["mock_test_class"]

def test_save_model(mock_train_pr):
    mock_train_pr.save_model = MagicMock()
    mock_train_pr.save_model("mock_filename")
    mock_train_pr.save_model.assert_called_once_with("mock_filename")
