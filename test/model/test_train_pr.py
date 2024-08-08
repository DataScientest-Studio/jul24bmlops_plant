import pytest
from unittest.mock import MagicMock
import tensorflow as tf
from train_model import TrainPR

@pytest.fixture
def mock_model():
    mock = MagicMock(spec=tf.keras.Model)
    mock.image_size = (180, 180)
    mock.batch_size = 32
    mock.base_learning_rate = 0.0001
    mock.fine_tune_at = 100
    mock.initial_epochs = 10
    mock.fine_tune_epochs = 10
    return mock

def test_init_with_model_path(mocker, mock_model):
    # Mock the load_model function to return the mock model
    mocker.patch('tensorflow.keras.models.load_model', return_value=mock_model)
    
    # Instantiate the TrainPR class, which should load the model
    train_pr = TrainPR(model_path="dummy_path")
    
    # Check that the model was set correctly
    assert train_pr.model == mock_model
    assert train_pr.image_size == (180, 180)
    assert train_pr.batch_size == 32
    assert train_pr.base_learning_rate == 0.0001
    assert train_pr.fine_tune_at == 100
    assert train_pr.initial_epochs == 10
    assert train_pr.fine_tune_epochs == 10

def test_init_with_kwargs():
    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    
    assert train_pr.model is None
    assert train_pr.image_size == (224, 224)
    assert train_pr.batch_size == 64
    assert train_pr.base_learning_rate == 0.00005
    assert train_pr.fine_tune_at == 50
    assert train_pr.initial_epochs == 5
    assert train_pr.fine_tune_epochs == 15

def test_update_hyperparameters():
    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    train_pr.update_hyperparameters(batch_size=128, fine_tune_at=75)

    assert train_pr.batch_size == 128
    assert train_pr.fine_tune_at == 75
    assert train_pr.image_size == (224, 224)  # Unchanged
    assert train_pr.base_learning_rate == 0.00005  # Unchanged

def test_build_model(mocker, mock_model):
    mocker.patch('tensorflow.keras.applications.MobileNetV2')
    mocker.patch('tensorflow.keras.Model', return_value=mock_model)

    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    train_pr.build_model(num_classes=5)

    assert train_pr.model == mock_model
    tf.keras.Model.assert_called_once()

def test_train_model(mocker, mock_model):
    mocker.patch.object(mock_model, 'fit', return_value=MagicMock())
    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    train_pr.model = mock_model

    history, history_fine = train_pr.train_model()

    assert mock_model.fit.call_count == 2

def test_save_model(mocker, mock_model):
    mocker.patch.object(mock_model, 'save')
    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    train_pr.model = mock_model

    train_pr.save_model("dummy_path")

    mock_model.save.assert_called_once_with("dummy_path")
