import os
import sys
import argparse
import logging
import json
import time
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa

# python bin/test.py --num_workers=20  --device_ids='2' weights weights/dev.csv weights/dev_test.csv

parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('model_path', default=None, metavar='MODEL_PATH', type=str,
                    help="Path to the trained models")
parser.add_argument('in_csv_path', default=None, metavar='IN_CSV_PATH',
                    type=str, help="Path to the input image path in csv")
parser.add_argument('out_csv_path', default=None, metavar='OUT_CSV_PATH',
                    type=str, help="Path to the ouput predictions in csv")
parser.add_argument('--num_workers', default=1, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")


def get_label(cfg, output):
    if cfg.criterion == "BCE":
        prob = torch.sigmoid(output)
        return prob.ge(0.5).float()
    elif cfg.criterion == "CE":
        return torch.argmax(output, dim=1)


def test_epoch(cfg, args, model, dataloader, out_csv_path):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)

    with open(cfg.train_csv) as f:
        test_header = f.readline().strip('\n').split(',')

    with open(out_csv_path, 'w') as f:
        f.write(','.join(test_header) + '\n')
        for step in range(steps):
            image, path = next(dataiter)
            image = image.to(device)
            output = model(image)

            # prob = torch.sigmoid(output)

            pred = Variable(torch.Tensor(image.size()[0], len(output)).type_as(output[0].data))
            for index in range(len(output)):
                # pred_batch = [str(value) for value in value]
                prob = F.softmax(output[index])
                pred[:, index] = prob[:, 1]

            for index in range(len(path)):
                pred_batch = ",".join(str(value) for value in pred[index].tolist())
                result = path[index] + ',' + pred_batch
                f.write(result + '\n')
                logging.info('{}, Image : {}, Prob : {}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), path[index], pred_batch))


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

    dataloader_test = DataLoader(
        ImageDataset(args.in_csv_path, cfg, mode='test'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    test_epoch(cfg, args, model, dataloader_test, args.out_csv_path)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
