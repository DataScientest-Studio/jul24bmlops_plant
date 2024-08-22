import os
from pathlib import Path
from enum import Enum

def save_image(image_data: bytes, label: str, filename: str):
     directory = Path(f"data/{label}")
     directory.mkdir(parents=True, exist_ok=True)
     file_path = directory / filename
     with open(file_path, "wb") as f:
          f.write(image_data)
     return str(file_path)


class ClassLabels(str, Enum):
     Apple_Apple_scab = "Apple___Apple_scab"
     Apple_Black_rot = "Apple___Black_rot"
     Apple_Cedar_apple_rust = "Apple___Cedar_apple_rust"
     Apple_healthy = "Apple___healthy"
     Background_without_leaves = "Background_without_leaves"
     Black_grass = "Black-grass"
     Blueberry_healthy = "Blueberry___healthy"
     Charlock = "Charlock"
     Cherry_Powdery_mildew = "Cherry___Powdery_mildew"
     Cherry_healthy = "Cherry___healthy"
     Cleavers = "Cleavers"
     Common_Chickweed = "Common Chickweed"
     Common_wheat = "Common wheat"
     Corn_Cercospora_leaf_spot_Gray_leaf_spot = "Corn___Cercospora_leaf_spot Gray_leaf_spot"
     Corn_Common_rust = "Corn___Common_rust"
     Corn_Northern_Leaf_Blight = "Corn___Northern_Leaf_Blight"
     Corn_healthy = "Corn___healthy"
     Fat_Hen = "Fat Hen"
     Grape_Black_rot = "Grape___Black_rot"
     Grape_Esca_Black_Measles = "Grape___Esca_(Black_Measles)"
     Grape_Leaf_blight_Isariopsis_Leaf_Spot = "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
     Grape_healthy = "Grape___healthy"
     Loose_Silky_bent = "Loose Silky-bent"
     Maize = "Maize"
     Orange_Haunglongbing_Citrus_greening = "Orange___Haunglongbing_(Citrus_greening)"
     Peach_Bacterial_spot = "Peach___Bacterial_spot"
     Peach_healthy = "Peach___healthy"
     Pepper_bell_Bacterial_spot = "Pepper,_bell___Bacterial_spot"
     Pepper_bell_healthy = "Pepper,_bell___healthy"
     Potato_Early_blight = "Potato___Early_blight"
     Potato_Late_blight = "Potato___Late_blight"
     Potato_healthy = "Potato___healthy"
     Raspberry_healthy = "Raspberry___healthy"
     Scentless_Mayweed = "Scentless Mayweed"
     Shepherds_Purse = "Shepherds Purse"
     Small_flowered_Cranesbill = "Small-flowered Cranesbill"
     Soybean_healthy = "Soybean___healthy"
     Squash_Powdery_mildew = "Squash___Powdery_mildew"
     Strawberry_Leaf_scorch = "Strawberry___Leaf_scorch"
     Strawberry_healthy = "Strawberry___healthy"
     Sugar_beet = "Sugar beet"
     Tomato_Bacterial_spot = "Tomato___Bacterial_spot"
     Tomato_Early_blight = "Tomato___Early_blight"
     Tomato_Late_blight = "Tomato___Late_blight"
     Tomato_Leaf_Mold = "Tomato___Leaf_Mold"
     Tomato_Septoria_leaf_spot = "Tomato___Septoria_leaf_spot"
     Tomato_Spider_mites_Two_spotted_spider_mite = "Tomato___Spider_mites Two-spotted_spider_mite"
     Tomato_Target_Spot = "Tomato___Target_Spot"
     Tomato_Tomato_Yellow_Leaf_Curl_Virus = "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
     Tomato_Tomato_mosaic_virus = "Tomato___Tomato_mosaic_virus"
     Tomato_healthy = "Tomato___healthy"



CLASS_NAMES = [
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