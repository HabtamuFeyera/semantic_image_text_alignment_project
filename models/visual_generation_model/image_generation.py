# semantic_image_text_alignment_project/models/visual_generation_model/image_generation.py

import cv2
import numpy as np

def generate_image(width, height):
    """
    Generate a blank white image with specified width and height.
    
    Args:
    - width (int): Width of the image.
    - height (int): Height of the image.
    
    Returns:
    - image (numpy.ndarray): Generated image.
    """
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White image
    return image

def add_text_to_image(image, text, font_path, font_size, position):
    """
    Add text to the image.
    
    Args:
    - image (numpy.ndarray): Input image.
    - text (str): Text to be added to the image.
    - font_path (str): Path to the font file.
    - font_size (int): Font size.
    - position (tuple): Position to place the text (x, y).
    
    Returns:
    - image_with_text (numpy.ndarray): Image with text added.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)  # Black color
    thickness = 2
    font_scale = font_size / 10
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image

