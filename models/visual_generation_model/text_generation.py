# semantic_image_text_alignment_project/models/visual_generation_model/text_generation.py

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_text_image(text, width, height, font_path, font_size):
    """
    Generate an image with the specified text.
    
    Args:
    - text (str): Text to be displayed.
    - width (int): Width of the image.
    - height (int): Height of the image.
    - font_path (str): Path to the font file.
    - font_size (int): Font size.
    
    Returns:
    - image (numpy.ndarray): Generated image.
    """
    # Generate a blank white image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add text to the image
    font = ImageFont.truetype(font_path, font_size)
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    text_width, text_height = draw.textsize(text, font=font)
    text_position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(text_position, text, font=font, fill=(0, 0, 0))
    
    return np.array(image_pil)

