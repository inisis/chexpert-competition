#!/usr/bin/env python
# coding=utf-8

import cv2
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, label_path, mode='train'):
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode 
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = header[1:]
            for line in f:
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                self._image_paths.append(image_path)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx])
        h, w , _ = image.shape 
        return (h, w)

def test_epoch(dataloader):
    steps = len(dataloader)
    dataiter = iter(dataloader)

    fig = plt.figure()

    for step in range(steps):
        h, w = next(dataiter)
        plt.scatter(h, w, c='k')
        print(step)
        
    plt.xlabel('height',fontsize=18,labelpad=18.8)
    plt.ylabel('width',fontsize=18,labelpad=12.5)
    plt.savefig("dataset_distribution.png")
    plt.show()

def main():

    #python src/<path-to-prediction-program> <input-data-csv-filename> <output-prediction-csv-path>

    infile = sys.argv[1]

    print(infile)

    dataloader_test = DataLoader(
        ImageDataset(infile),
        batch_size = 64, num_workers = 24,
        drop_last = False, shuffle = False)

    test_epoch(dataloader_test)


if __name__ == '__main__':
    main()
