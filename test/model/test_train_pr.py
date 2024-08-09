import pytest
from unittest.mock import MagicMock, patch
import tensorflow as tf
import numpy as np
from models.train_model import TrainPR

@pytest.fixture
def mock_model():
    mock = MagicMock(spec=tf.keras.Model)
    mock.image_size = (180, 180)
    mock.batch_size = 32
    mock.base_learning_rate = 0.0001
    mock.fine_tune_at = 100
    mock.initial_epochs = 10
    mock.fine_tune_epochs = 10
    mock.class_names = ['class1', 'class2', 'class3']
    return mock

def test_init_with_model_path(mocker, mock_model):
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

def test_load_data(mocker):
    mock_image_dataset = MagicMock()
    mock_image_dataset.class_names = ['class1', 'class2', 'class3']

    # Mock the image_dataset_from_directory function
    mocker.patch('tensorflow.keras.utils.image_dataset_from_directory', return_value=mock_image_dataset)

    # Mock the cardinality function
    mock_cardinality = mocker.patch('tensorflow.data.experimental.cardinality', return_value=10)

    # Mock the take and skip methods of the dataset
    mock_image_dataset.take.return_value = mock_image_dataset
    mock_image_dataset.skip.return_value = mock_image_dataset

    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    train_pr.model = MagicMock(spec=tf.keras.Model)

    train_pr.load_data('dummy_path')

    # Assert that the datasets are set correctly
    assert train_pr.train_ds == mock_image_dataset
    assert train_pr.val_ds == mock_image_dataset
    assert train_pr.test_ds == mock_image_dataset

    # Assert that model.class_names is set correctly
    assert train_pr.model.class_names == ['class1', 'class2', 'class3']

    # Assert that cardinality was called
    mock_cardinality.assert_called_once_with(mock_image_dataset)

def test_preprocess(mocker):
    mock_dataset = MagicMock()
    mock_dataset.cache.return_value = mock_dataset
    mock_dataset.prefetch.return_value = mock_dataset
    
    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    train_pr.train_ds = mock_dataset
    train_pr.val_ds = mock_dataset
    train_pr.test_ds = mock_dataset

    train_pr.preprocess()

    assert train_pr.train_ds.cache.called
    assert train_pr.val_ds.cache.called
    assert train_pr.test_ds.cache.called
    assert train_pr.train_ds.prefetch.called
    assert train_pr.val_ds.prefetch.called
    assert train_pr.test_ds.prefetch.called

def test_build_model(mocker, mock_model):
    mocker.patch('tensorflow.keras.applications.MobileNetV2')
    mocker.patch('tensorflow.keras.Model', return_value=mock_model)

    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    train_pr.model = mock_model  # Manually setting the model to have the mock methods
    train_pr.model.class_names = ['class1', 'class2', 'class3']
    train_pr.build_model()

    assert train_pr.model == mock_model
    tf.keras.Model.assert_called_once()

def test_train_model(mocker, mock_model):
    mock_history = MagicMock()
    mocker.patch.object(mock_model, 'fit', return_value=mock_history)
    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    train_pr.model = mock_model

    history, history_fine = train_pr.train_model()

    assert mock_model.fit.call_count == 2

def test_predict(mocker):
    # Mock the model's prediction
    mock_model = MagicMock(spec=tf.keras.Model)
    mock_model.return_value = tf.convert_to_tensor([[0.1, 0.7, 0.2]])  # Simulate softmax output
    mock_model.class_names = ['class1', 'class2', 'class3']

    # Mock the test dataset
    mock_test_dataset = [(
        tf.convert_to_tensor(np.random.random((1, 180, 180, 3)), dtype=tf.float32),
        tf.convert_to_tensor([1], dtype=tf.int32)
    )]
    mock_test_ds = MagicMock()
    mock_test_ds.__iter__.return_value = iter(mock_test_dataset)

    # Create TrainPR instance and set model and test_ds
    train_pr = TrainPR(image_size=(180, 180))
    train_pr.model = mock_model
    train_pr.test_ds = mock_test_ds

    # Call the predict method
    predicted_classes, test_classes = train_pr.predict()

    # Assertions
    assert np.array_equal(predicted_classes, np.array([1]))
    assert np.array_equal(test_classes, np.array([1]))

def test_save_model(mocker, mock_model):
    mocker.patch.object(mock_model, 'save')
    train_pr = TrainPR(image_size=(224, 224), batch_size=64, base_learning_rate=0.00005, fine_tune_at=50, initial_epochs=5, fine_tune_epochs=15)
    train_pr.model = mock_model

    train_pr.save_model("dummy_path")

    mock_model.save.assert_called_once_with("dummy_path")
