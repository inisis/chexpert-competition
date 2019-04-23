import os
import sys
import argparse
import logging
import json
import time
import numpy as np
from easydict import EasyDict as edict
import torch
from torch import topk
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib.pyplot import imshow, show
from PIL import Image
import skimage.transform

import cv2

from torchvision import models, transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa

# python bin/test.py --num_workers=20  --device_ids='2' weights weights/dev.csv weights/dev_test.csv

parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('model_path', default=None, metavar='MODEL_PATH', type=str,
                    help="Path to the trained models")
parser.add_argument('in_csv_path', default=None, metavar='IN_CSV_PATH',
                    type=str, help="Path to the input image path in csv")
parser.add_argument('--num_workers', default=1, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")


class save_features():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


def get_cam(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam) * 255
    return [cam_img]


display_transform = transforms.Compose([
   transforms.Resize((224,224))])


def test_epoch(cfg, args, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    final_layer = model.module.backbone._modules.get('layer4')

    activated_features = save_features(final_layer)

    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)

    for step in range(steps):
        image, path = next(dataiter)
        image = image.to(device)
        output = model(image)

        print(path[0])

        for index in range(len(output)):
            pred_probabilities = F.softmax(output[index], dim=1).data.squeeze()
            activated_features.remove()

            weight_softmax_params = list(model.module.fc_0.weight.data.cpu().data.numpy())

            weight_softmax = np.array(weight_softmax_params)

            class_idx = topk(pred_probabilities, 1)[1].int()

            overlay = get_cam(activated_features.features, weight_softmax, class_idx)

            pixel_value = (np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0)))

            pixel_value += cfg.pixel_mean

            mask = skimage.transform.resize(overlay[0], image.shape[2:4])

            mask_rgb = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite("image_" + path[0] + "_task" + str(index) + "class" + str(class_idx.cpu().detach().numpy()) + "image_origin.jpg", pixel_value, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            img_mix = cv2.addWeighted(pixel_value, 1, mask_rgb.astype(np.float32), 0.5, 0)

            cv2.imwrite("image_" + path[0] + "_task" + str(index) + "class" + str(class_idx.cpu().detach().numpy()) + ".jpg", img_mix, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def run(args):
    with open(args.model_path+'cfg.json') as f:
        cfg = edict(json.load(f))

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
            .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg)
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    ckpt_path = os.path.join(args.model_path, 'best.ckpt')
    ckpt = torch.load(ckpt_path)
    model.module.load_state_dict(ckpt['state_dict'])
    from torchsummary import summary
    summary(model.to(device), (3, cfg.height, cfg.width))

    dataloader_test = DataLoader(
        ImageDataset(args.in_csv_path, cfg, mode='test'),
        batch_size=1, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    test_epoch(cfg, args, model, dataloader_test)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
