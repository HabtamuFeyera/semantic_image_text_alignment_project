import torch
from models.text_module import TextUnderstandingModule
from models.image_module import ImageGenerationModule
from utils.preprocess import preprocess_text

# Load trained models
text_module = TextUnderstandingModule(input_size, hidden_size, num_layers, num_classes)
text_module.load_state_dict(torch.load('saved_models/text_module.pth'))
text_module.eval()

image_module = ImageGenerationModule(latent_dim, img_channels, img_size)
image_module.load_state_dict(torch.load('saved_models/image_module.pth'))
image_module.eval()

# Define input text
input_text = "Your input text here..."

# Preprocess input text
# Replace this with your actual text preprocessing logic if needed
text_inputs = preprocess_text(input_text)

# Convert input text to tensor and reshape if necessary
text_inputs = torch.tensor(text_inputs).unsqueeze(0)  # Add batch dimension if needed

# Generate storyboard
with torch.no_grad():
    text_features = text_module(text_inputs)
    generated_images = image_module(text_features)

