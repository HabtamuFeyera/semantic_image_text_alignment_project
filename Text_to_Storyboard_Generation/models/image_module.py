import torch
import torch.nn as nn

class ImageGenerationModule(nn.Module):
    def __init__(self, latent_dim, img_channels=3, img_size=64):
        super(ImageGenerationModule, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # Define the generator architecture
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, img_size * img_size * img_channels),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.generator(z)
        img = img.view(-1, self.img_channels, self.img_size, self.img_size)
        return img
