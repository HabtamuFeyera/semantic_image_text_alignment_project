import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class AdvertisementDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for project_folder in os.listdir(self.root_dir):
            project_path = os.path.join(self.root_dir, project_folder)
            if os.path.isdir(project_path):
                samples.append(project_path)
        return samples

    def _load_assets(self, project_path):
        landing_image = None
        endframe_image = None
        other_assets = []

        for asset_file in os.listdir(project_path):
            asset_path = os.path.join(project_path, asset_file)
            if os.path.isfile(asset_path):
                if 'landing' in asset_file.lower():
                    landing_image = Image.open(asset_path)
                elif 'endframe' in asset_file.lower():
                    endframe_image = Image.open(asset_path)
                else:
                    other_assets.append(Image.open(asset_path))

        return landing_image, endframe_image, other_assets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        project_path = self.samples[idx]
        landing_image, endframe_image, other_assets = self._load_assets(project_path)
        # Implement preprocessing of assets (resize, normalize, etc.)
        return landing_image, endframe_image, other_assets  # Return preprocessed assets for the project
