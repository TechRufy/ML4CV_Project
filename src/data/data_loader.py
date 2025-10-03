import glob
import os

import pandas as pd
import torch

from torch.utils.data import Dataset

from torchvision.io import read_image


class StreetHazardsDataset(Dataset):
    dataset = "StreetHazards"

    def __init__(self, split):
        if split in ["train", "test", "val"]:
            self.label_directory = f"./data/StreetHazards/{split}/annotations"
            self.img_dir = f"./data/StreetHazards/{split}/images"
        else:
            assert "split not valid, must be one of ['train', 'test', 'val']"
        self.img_files = []
        self.img_labels = []

        for filename in glob.iglob(self.img_dir + "/**/*.*", recursive=True):
            self.img_files.append(filename)

        for filename in glob.iglob(self.label_directory + "/**/*.*", recursive=True):
            self.img_labels.append(filename)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        GT_path = self.img_labels[idx]

        image = read_image(img_path)[:-1, :, :].float()

        GT = read_image(GT_path)

        return image, GT
