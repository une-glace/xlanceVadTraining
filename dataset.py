import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DummyVADDataset(Dataset):
    def __init__(self, length=1000):
        """
        A dummy dataset generator for VAD training.
        In real usage, this should load audio files and labels.
        """
        self.length = length
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 1. Simulate Input: 80-dim Mel Spectrogram, 200 frames (approx 2 sec)
        # Shape: [80, 200]
        feature = torch.randn(80, 200)
        
        # 2. Simulate Label: 0 or 1 for each frame (Time dimension)
        # Note: Since our model has a MaxPool (stride=2), the output time dim is halved.
        # We need to provide labels that match the model's output resolution (100 frames).
        label = torch.randint(0, 2, (100, 1)).float()
        
        return feature, label

def get_dataloader(batch_size=32):
    dataset = DummyVADDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
