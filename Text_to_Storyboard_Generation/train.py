import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import AdvertisementDataset
from models.text_module import TextUnderstandingModule
from models.image_module import ImageGenerationModule
from utils.preprocess import preprocess_image, preprocess_text
from utils.evaluation import evaluate_model

# Import configurations
from config import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dataset and dataloader
dataset = AdvertisementDataset(root_dir='data/Preprocessed_Assets')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize text understanding module
text_module = TextUnderstandingModule(input_size, hidden_size, num_layers, num_classes).to(device)

# Initialize image generation module
image_module = ImageGenerationModule(latent_dim, img_channels, img_size).to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(list(text_module.parameters()) + list(image_module.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    text_module.train()
    image_module.train()
    total_loss = 0.0

    for landing_images, endframe_images, other_assets in dataloader:
        # Move data to device
        landing_images = landing_images.to(device)
        endframe_images = endframe_images.to(device)
        other_assets = [asset.to(device) for asset in other_assets]

        # Preprocess text
        # Replace the following line with your actual text preprocessing logic if needed
        text_inputs = torch.randn(batch_size, input_size).to(device)

        # Forward pass
        text_features = text_module(text_inputs)
        generated_images = image_module(text_features)

        # Compute loss
        loss = criterion(generated_images, endframe_images)
        total_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print epoch statistics
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Evaluation
    # Example: You can add evaluation code here if needed

# Save trained models
torch.save(text_module.state_dict(), 'saved_models/text_module.pth')
torch.save(image_module.state_dict(), 'saved_models/image_module.pth')
