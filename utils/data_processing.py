# /home/habte/semantic_image_text_alignment_project/utils/data_processing.py

import os
from PIL import Image
import numpy as np

def preprocess_assets(assets_folder, output_folder, target_size=(224, 224)):
    """
    Preprocesses assets by resizing images and saving them to the output folder.

    Args:
    - assets_folder (str): Path to the folder containing the assets.
    - output_folder (str): Path to the folder where preprocessed assets will be saved.
    - target_size (tuple): Target size for resizing images. Default is (224, 224).
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through subfolders
    for project_folder in os.listdir(assets_folder):
        project_path = os.path.join(assets_folder, project_folder)
        
        # Locate key images
        landing_image_path = os.path.join(project_path, 'landing.jpg')
        endframe_image_path = os.path.join(project_path, 'endframe.jpg')
        
        # Preprocess key images
        landing_image = preprocess_image(landing_image_path, target_size)
        endframe_image = preprocess_image(endframe_image_path, target_size)

        # Save preprocessed images
        output_project_folder = os.path.join(output_folder, project_folder)
        if not os.path.exists(output_project_folder):
            os.makedirs(output_project_folder)
        
        landing_output_path = os.path.join(output_project_folder, 'landing.jpg')
        endframe_output_path = os.path.join(output_project_folder, 'endframe.jpg')
        
        save_image(landing_image, landing_output_path)
        save_image(endframe_image, endframe_output_path)

def preprocess_image(image_path, target_size):
    """
    Preprocesses an individual image by resizing it and normalizing pixel values.

    Args:
    - image_path (str): Path to the image file.
    - target_size (tuple): Target size for resizing the image.

    Returns:
    - numpy.ndarray: Preprocessed image as a NumPy array.
    """
    img = Image.open(image_path)
    img = img.resize(target_size)  # Resize to a consistent size
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

def save_image(image, output_path):
    """
    Saves an image to the specified output path.

    Args:
    - image (numpy.ndarray): Image as a NumPy array.
    - output_path (str): Path where the image will be saved.
    """
    Image.fromarray((image * 255).astype(np.uint8)).save(output_path)
