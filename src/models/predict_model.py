"""
script that performs predictions on the target (test) set using the (target) model and prints confusion matrix & classification report
""" 
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

cls_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

assert len(cls_names) == 38

def prdct(model_file_path: str, dataset: tf.data.Dataset):
    """
    Predicts classes for the target (test) dataset using the trained model

    Returns:
        pred_cls (numpy.array): Array of predicted classes.
        true_cls (numpy.array): Array of true classes.
    """
    true_cls = np.array([])
    pred_cls = np.array([])

    model = tf.keras.models.load_model(model_file_path)

    for x, y in dataset:
        pred_cls = np.concatenate([pred_cls, np.argmax(model(x, training=False), axis=-1)]).astype(int)
        true_cls = np.concatenate([true_cls, y.numpy()]).astype(int)
    
    return true_cls, pred_cls

def show_clsf_rprt(model_file_path: str, dataset: tf.data.Dataset) -> None:
    """
    generates (prints) classification report for all the true classes
    """
    true_cls, pred_cls = prdct(model_file_path, dataset)
    lbl_true_cls = np.array(cls_names)[true_cls]
    lbl_pred_cls = np.array(cls_names)[pred_cls]
    print(classification_report(lbl_true_cls, lbl_pred_cls))

def show_conf_mtrx(model_file_path: str, dataset: tf.data.Dataset) -> None:
    """
    generates (prints) the confusion matrix
      x-axis (rows): true classes
      y-axis (columns): predicted classes
    """
    true_cls, pred_cls = prdct(model_file_path, dataset)
    print(confusion_matrix(true_cls, pred_cls))