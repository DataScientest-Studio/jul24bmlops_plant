from tkinter import HIDDEN
import utils
from PIL import Image
import streamlit as st

import base64
import time
import os
import re
from pathlib import Path
from typing import BinaryIO, Tuple
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# packages imported for classical model
from skimage.feature import hog
import xgboost as xgb
import cv2


class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Background_without_leaves",
    "Black-grass",
    "Blueberry___healthy",
    "Charlock",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Fat Hen",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Loose Silky-bent",
    "Maize",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Sugar beet",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

luigi = {
    "model_name": "TL_180px_32b_20e_model.keras",
    "color_mode": "RGB",
    "target_size": (180, 180),
    "interpolation": "BILINEAR",
    "keep_aspect_ratio": False,
}

# Mapping model types to their configurations
model_list = {
    "Transfer Learning": luigi,
}


def prediction_home():
    """
    Function to perform image prediction and display results.

    Returns:
        None
    """
    # Initialize session state for previous value if not already initialized
    if "previous_mod_sel_value" not in st.session_state:
        st.session_state.previous_mod_sel_value = None

    if "previous_up_img_value" not in st.session_state:
        st.session_state.previous_up_img_value = None

    if "model_value" not in st.session_state:
        st.session_state.model_value = None

    st.markdown("# Prediction 🍃")
    st.subheader("1. Choose which model you want to use")

    csb1, _, _ = st.columns(3)
    with csb1:
        selected_model = st.selectbox("Select a model to load:",
                                      ["Please select a model..."] +
                                      list(model_list.keys()),
                                      key="model_select_box", label_visibility=HIDDEN
                                      )

        # Conditional content based on the selection
        if selected_model != "Please select a model...":
            if selected_model != st.session_state.previous_mod_sel_value:
                # Update the previous value in the session state
                st.session_state.previous_mod_sel_value = selected_model
                st.session_state.previous_up_img_value = False

                model_file = model_list[selected_model]["model_name"]

                st.write(f"Loading model: {model_file}")

                if selected_model == "Machine Learning (XGBClassifier)":
                    model = utils.load_classical_model(
                        "../models/" + model_file)
                else:
                    model = utils.load_model_with_progress(
                        "../models/" + model_file)

                st.session_state.model_value = model

                st.success(f"Model {selected_model} loaded successfully!")
                st.write(
                    "Now you can use the model for predictions or further analysis:")

            st.write("")
            st.subheader("Upload an image")
            st.markdown(
                "*Note: to pursue a purposeful usage, do not load impractical images.*")

            image, image_valid = utils.upload_image()
            st.session_state.previous_up_img_value = image_valid

            img_info = Image.open(image)
            file_details = f"""
                Name: {image.name}
                Type: {img_info.format}
                Size: {img_info.size}
            """

            st.write("")
            st.subheader("Results")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original image ...")

                st.image(img_info, width=150)
                if st.session_state.previous_up_img_value:
                    st.caption(file_details)

            with col2:
                with st.container():
                    st.subheader("... is probably :")

                    # Add here the prediction model result.
                    if st.session_state.previous_up_img_value:
                        if selected_model == "Machine Learning (XGBClassifier)":
                            result = utils.classical_ml_predict(
                                st.session_state.model_value, img_info)
                            confidence = ""
                        else:
                            image_array = utils.preprocess_image(
                                img_info, model_list[selected_model])
                            result, confidence = utils.predict(
                                st.session_state.model_value, image_array)

                        result = "web/img/classes/" + result + "_leaf.png"
                        img_result = Image.open(result)
                        st.image(img_result, width=150)
                        st.write(confidence)




def preprocess_image(pil_image: str, data: object) -> np.array:
    """
    Load and preprocess an image to be suitable for model prediction.

    Parameters:
    - img_path (str): The path to the image.
    - target_size (tuple): The target size of the image (height, width).

    Returns:
    - image_array (np.array): Preprocessed image array.
    """
    # Convert to RGB if it's not already
    if data["color_mode"] == "RGB":
        pil_image = pil_image.convert('RGB')
    else:
        pil_image = pil_image.convert('L')

    # Resize the image
    target_size = data["target_size"]
    interpolation = Image.BILINEAR  # PIL interpolation methods

    keep_aspect_ratio = data["keep_aspect_ratio"]

    if not keep_aspect_ratio:
        resized_image = pil_image.resize(target_size, interpolation)
    else:
        # Optionally handle aspect ratio if needed (manual implementation required)
        resized_image = pil_image  # Placeholder for actual aspect ratio handling

    # Convert to a NumPy array
    image_array = np.array(resized_image)

    # Ensure the image is in the expected shape (height, width, channels)
    image_array = tf.convert_to_tensor(image_array, dtype=tf.float32)

    # Add a batch dimension if required (model expects batched input)
    image_array = tf.expand_dims(image_array, axis=0)

    return image_array


def predict(model: tf.keras.Model, image_array: np.array) -> np.array:
    """
    Predict the class of an image using the loaded model.

    Parameters:
    - model: The loaded Keras model.
    - img_array (np.array): Preprocessed image array for prediction.

    Returns:
    - prediction: The predicted result.
    """
    # prediction = model.predict(image_array)
    predicted_classes = np.array([])
    predicted_classes = np.concatenate([predicted_classes, np.argmax(
        model(image_array, training=False), axis=-1)]).astype(int)
    c_predicted_class = np.array(class_names)[predicted_classes][0]
    confidence = np.max(model(image_array, training=False), axis=-1).item()
    confidence = f"{confidence:.2%}"

    return c_predicted_class, confidence


def load_model_with_progress(model_path) -> tf.keras.Model:
    # Create a progress bar
    progress_bar = st.progress(0)

    # Simulate a loading process with incremental updates
    for i in range(100):
        # Update progress bar
        time.sleep(0.01)  # Simulate some aspect of loading
        progress_bar.progress(i + 1)

    # Load your model (assuming the model is saved in the same directory)
    model = tf.keras.models.load_model(model_path)

    # Complete the progress bar
    progress_bar.progress(100)
    progress_bar.empty()

    return model
