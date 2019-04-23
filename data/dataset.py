import numpy as np
from torch.utils.data import Dataset
import cv2

np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
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
                labels = list(map(int, fields[1:]))
                self._image_paths.append(image_path)
                self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx]).astype(np.float32)
        h, w, c = image.shape
        fx, fy = float(self.cfg.width) / w, float(self.cfg.height) / h
        image = cv2.resize(image, None, None, fx=fx, fy=fy,
                           interpolation=cv2.INTER_LINEAR)
        # normalization
        image -= self.cfg.pixel_mean
        # vgg and resnet do not use pixel_std, densenet and inception use.
        if self.cfg.use_pixel_std:
            image /= self.cfg.pixel_std
        # normal image:  H x W x C
        # torch image:   C X H X W
        image = image.transpose((2, 0, 1))

        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
