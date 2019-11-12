import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image

np.random.seed(0)


class CSVDataset(Dataset):
    def __init__(self, label_path, cfg, mode='train', transform=None):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.transform = transform
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                    {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'},]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = header[1:]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                for index, value in enumerate(fields[1:]):
                    if index == 1 or index == 3:
                        labels.append(self.dict[1].get(value))
                    elif index == 0 or index == 2 or index == 4:
                        labels.append(self.dict[0].get(value))
                labels = list(map(int, labels))
                if (labels[0] == 1 or labels[2] == 1) and mode == 'train':
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                else:
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def _fix_ratio(self, image):
        h, w, c = image.shape

        if h >= w:
            ratio = h * 1.0 / w
            h_ = self.cfg.long_side
            w_ = round(h_ / ratio)
        else:
            ratio = w * 1.0 / h
            w_ = self.cfg.long_side
            h_ = round(w_ / ratio)

        image = cv2.resize(image, dsize=(w_, h_),
                           interpolation=cv2.INTER_LINEAR)
        image = np.pad(
            image,
            ((0, self.cfg.long_side-h_), (0, self.cfg.long_side-w_), (0, 0)),
            mode='constant', constant_values=self.cfg.pixel_mean
        )

        return image

    def __getitem__(self, idx):
        pil_image = Image.open(self._image_paths[idx])
        if self.transform is not None:
           pil_image = self.transform(pil_image)
        image = cv2.cvtColor(np.array(pil_image),cv2.COLOR_RGB2BGR).astype(np.float32)

        # image = cv2.imread(self._image_paths[idx]).astype(np.float32)

        if self.cfg.fix_ratio:
            image = self._fix_ratio(image)
        else:
            image = cv2.resize(image, dsize=(self.cfg.width, self.cfg.height),
                               interpolation=cv2.INTER_LINEAR)
        
        # normalization
        image -= self.cfg.pixel_mean
        # vgg and resnet do not use pixel_std, densenet and inception use.
        if self.cfg.use_pixel_std:
            image /= self.cfg.pixel_std
        # normal image tensor :  H x W x C
        # torch image tensor :   C X H X W
        image = image.transpose((2, 0, 1))
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
