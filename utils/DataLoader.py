import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_directory, img_dir):
        self.annotations_directory = annotations_directory
        self.img_dir = img_dir
        self.img_files = []
        for name in os.listdir(self.img_dir):
            self.img_files.append(name)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        GT_path = os.path.join(self.annotations_directory, self.img_files[idx])
        image = read_image(img_path)
        GT = read_image(GT_path)
        return image, GT
