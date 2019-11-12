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

fixed_size_ = 256

class ImageDataset(Dataset):
    def __init__(self, label_path, mode='train'):
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode 
        self.dict = {'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'}
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = header[1:]
            for line in f:
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                self._image_paths.append(image_path)
                self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def _fix_ratio(self, image):
        h, w, c = image.shape

        if h >= w:
            ratio = h * 1.0 / w
            h_ = fixed_size_
            w_ = round(h_ / ratio)
        else:
            ratio = w * 1.0 / h
            w_ = fixed_size_
            h_ = round(w_ / ratio)

        image = cv2.resize(image, dsize=(w_, h_),
                           interpolation=cv2.INTER_LINEAR)
        image = np.pad(
            image,
            ((0, fixed_size_ - h_), (0, fixed_size_ - w_), (0, 0)),
            mode='constant', constant_values = 128.0
        )

        return image
    
    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx]).astype(np.float32)
        image = self._fix_ratio(image)
        
        # normalization 
        image -= 128.0
        
        # normal image tensor :  H x W x C
        # torch image tensor :   C X H X W
        image = image.transpose((2, 0, 1))
        labels = np.array(self._labels[idx]).astype(np.float32)
            
        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))  

def get_pred(output):
    return torch.sigmoid(output.view(-1)).cpu().detach().numpy()

def test_epoch(model, dataloader, out_csv_path):
    torch.set_grad_enabled(False)
    model.eval()
    device = torch.device('cuda')
    steps = len(dataloader)
    dataiter = iter(dataloader)
    header = ['Study', 'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

    with open(out_csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for step in range(steps):
            image, path = next(dataiter)
            image = image.to(device)
            output = model(image)
            batch_size = len(path)
            pred = np.zeros((5, batch_size))

            # pred[0] = get_pred(output[2])
            # pred[1] = get_pred(output[5])
            # pred[2] = get_pred(output[6])
            # pred[3] = get_pred(output[8])
            # pred[4] = get_pred(output[10])
            
            pred[0] = get_pred(output[0])
            pred[1] = get_pred(output[1])
            pred[2] = get_pred(output[2])
            pred[3] = get_pred(output[3])
            pred[4] = get_pred(output[4])

            for i in range(batch_size):
                batch = ','.join(map(lambda x: '{:.16}'.format(x), pred[:, i]))
                result = path[i] + ',' + batch
                f.write(result + '\n')


def main():

    #python src/<path-to-prediction-program> <input-data-csv-filename> <output-prediction-csv-path>

    infile = sys.argv[1]
    outfile = sys.argv[2]

    print(infile)
    print(outfile)

    device = torch.device('cuda')
    model = torch.load("./src/competition.pth", map_location='cuda')
    model = model.to(device)

    dataloader_test = DataLoader(
        ImageDataset(infile, mode='test'),
        batch_size = 64, num_workers = 6,
        drop_last = False, shuffle = False)

    test_epoch(model, dataloader_test, outfile)

    test_df = pd.read_csv(outfile)                                                                                                                                                                                   

    #CheXpert-v1.0/{valid,test}/<PATIENT>/<STUDY>

    test_df.Study.str.split('/')

    def get_study(path):
        return path[0:path.rfind('/')]

    test_df['Study'] = test_df.Study.apply(get_study)

    study_df = test_df.groupby('Study').max().reset_index()

    study_df.to_csv(outfile,index=False)

if __name__ == '__main__':
    main()
