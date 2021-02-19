import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

class TextRecogDataset(Dataset):
    def __init__(self, items, encoder, augmentations):
        self.items = items
        self.encoder = encoder
        self.augmentations = augmentations
        
    def text_to_target(self, text):
        target = self.encoder.transform(list(text)) + 1
        return target
    
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        img_name = item["img_name"]
        label = item["label"]
        
        target = self.text_to_target(label)
        
        img = cv2.imread(img_name, 1)
        if self.augmentations:
            sample = self.augmentations(image=img)
            img = sample["image"]
        return img, torch.Tensor(target).int(), label