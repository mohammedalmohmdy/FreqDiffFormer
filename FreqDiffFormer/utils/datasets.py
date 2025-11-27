import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import csv

class SketchyDataset(Dataset):
    """
    Minimal dataset loader skeleton. Expects CSV with: image_path, sketch_path, label
    """
    def __init__(self, csv_file, root="", transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3 and row[0].strip().lower() != 'image_path':
                    self.samples.append(row[:3])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, sketch_path, label = self.samples[idx]
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        sketch = Image.open(os.path.join(self.root, sketch_path)).convert('L')
        if self.transform:
            # transform is a tuple (sketch_transform, image_transform)
            sketch_t, img_t = self.transform
            sketch = sketch_t(sketch)
            img = img_t(img)
        return sketch, img, int(label)
