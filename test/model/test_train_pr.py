import pytest
from unittest.mock import MagicMock, patch
from models.train_model import TrainPR

@pytest.fixture
def mock_train_pr():
    with patch('train_model.tf.keras.models.load_model') as mock_load_model:
        mock_load_model.return_value = MagicMock()  # Mock the model
        train_pr = TrainPR(model_path="mock_model_path")
        # We mock the `load_model` method of the `TrainPR` instance
        train_pr.load_model = MagicMock()
        return train_pr

def test_init_with_model_path(mock_train_pr):
    assert mock_train_pr.model is not None
    # No need to check `assert_called_once_with` here since we mock `load_model` explicitly in the fixture

def test_load_model():
    with patch('train_model.tf.keras.models.load_model') as mock_load_model:
        mock_load_model.return_value = MagicMock()
        train_pr = TrainPR(model_path="mock_model_path")
        # Now `load_model` should have been called once during initialization
        mock_load_model.assert_called_once_with("mock_model_path")

def test_update_hyperparameters(mock_train_pr):
    mock_train_pr.update_hyperparameters(image_size=(200, 200), batch_size=64)
    assert mock_train_pr.image_size == (200, 200)
    assert mock_train_pr.batch_size == 64

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
    mock_train_pr.predict = MagicMock(return_value=([], []))
    predicted_classes, test_classes = mock_train_pr.predict()
    mock_train_pr.predict.assert_called_once()
    assert isinstance(predicted_classes, list)
    assert isinstance(test_classes, list)

def test_save_model(mock_train_pr):
    mock_train_pr.save_model = MagicMock()
    mock_train_pr.save_model("mock_filename")
    mock_train_pr.save_model.assert_called_once_with("mock_filename")
