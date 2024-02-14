import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image by resizing and normalizing pixel values.
    
    Args:
    - image_path (str): Path to the image file.
    - target_size (tuple): Desired size for the image after resizing.
    
    Returns:
    - np.ndarray: Preprocessed image as a numpy array.
    """
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to range [0, 1]
    return img

def preprocess_text(text):
    """
    Preprocesses text data.
    
    Args:
    - text (str): Input text.
    
    Returns:
    - np.ndarray: Preprocessed text data.
    """
    # Tokenize text, encode using appropriate encoding technique, etc.
    # Example: Convert text to word embeddings or one-hot encoding
    # Implement your preprocessing logic here
    return processed_text
