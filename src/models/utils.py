import os
from typing import Tuple

import pandas as pd
from PIL import Image


# Helper functions
def image_data(filepath: str) -> Tuple[int, int, str, str]:
    """This function returns all the information related to the file passed in input.

    Args:
        filepath (str): The image file path.

    Returns:
        Tuple[int, int, str, str]: width as number in pixel, height as number in pixel, format as string (JPEG, PNG, GIF, etc.), mode as string (RGB, RGBA, etc.).
    """
    # open image
    image = Image.open(filepath)
    # Process the image here
    width, height = image.size  # size of image
    format = image.format  # JPEG, PNG, GIF
    mode = image.mode  # RGB, RGBA or others
    image.close()

    return width, height, format, mode


# Helper functions
def populate_dataset(dataset: dict, directory: str):
    """This function populate the passed dataset.

    Args:
        dataset (dict): The dataset that will be populated.
        directory (str): The directory where all the data is available.
    """
    # Loop through all the files and folders in the directory
    for folder_name in os.listdir(directory):
        if "___" in folder_name:
            plant, healthy = folder_name.split("___")
        else:
            plant = folder_name
            healthy = "healthy"

        if os.path.isdir(os.path.join(directory, folder_name)):
            for file_name in os.listdir(os.path.join(directory, folder_name)):
                file_path = os.path.join(directory, folder_name, file_name)
                width, height, format, mode = image_data(file_path)

                dataset["Plant"].append(plant)
                if healthy == "healthy":
                    dataset["Healthy"].append(1)
                else:
                    dataset["Healthy"].append(0)
                dataset["Illness"].append(healthy.replace(plant + "_", ""))
                dataset["Image_dir"].append(folder_name)
                dataset["Image_name"].append(file_name)
                dataset["Image_width"].append(width)
                dataset["Image_height"].append(height)
                dataset["Image_format"].append(format)
                dataset["Image_mode"].append(mode)


# The condition __name__ == "__main__" is used in a Python program to execute the code inside the if statement only
# when the program is executed directly by the Python interpreter.
# When the code in the file is imported as a module the code inside the if statement is not executed.
if __name__ == "__main__":
    # Assuming 'train' is the main directory containing all the categories
    dataset_dir = "data/raw"

    # Create a DataFrame with the following columns in the dataset
    # Columns name: Plant, Healthy, Illness, Image_name, Image_width, Image_height, Image_format, Image_mode
    data = {
        "Plant": [],
        "Healthy": [],
        "Illness": [],
        "Image_dir": [],
        "Image_name": [],
        "Image_width": [],
        "Image_height": [],
        "Image_format": [],
        "Image_mode": [],
    }

    populate_dataset(data, dataset_dir)

    # Create a sample DataFrame
    df = pd.DataFrame(data)

    # Export to CSV
    df.to_csv("data/export_train_dataframe.csv")
