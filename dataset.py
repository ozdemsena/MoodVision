import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, usage="Training", transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["Usage"] == usage]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emotion = int(self.data.iloc[idx]["emotion"])
        pixels = self.data.iloc[idx]["pixels"]
        pixels = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels)

        if self.transform:
            img = self.transform(img)

        return img, emotion

# Hızlı test
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(),  # zaten grayscale ama garanti olsun
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = FER2013Dataset(csv_file="fer2013.csv", usage="Training", transform=transform)
    print("Dataset length:", len(dataset))
    img, label = dataset[0]
    print("Image shape:", img.shape, "Label:", label)
