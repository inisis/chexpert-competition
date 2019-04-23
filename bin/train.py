import sys
import os
import argparse
import logging
import json
import time
from shutil import copyfile

from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.nn import DataParallel
from torch.autograd import Variable

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa
from utils.misc import lr_schedule  # noqa

# python bin/train.py --num_workers=20 --device_ids='7' \
# config/resnet18.json weights

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=1, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")
parser.add_argument('--resume', default=0, type=int, help="If resume from "
                    "previous run")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")


def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adadelta':
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adagrad':
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RMSprop':
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(cfg.optimizer))


def get_criterion(cfg):
    if cfg.criterion == 'BCE':
        assert len(cfg.num_classes) == 1
        assert cfg.num_classes[0] == 1
        return F.binary_cross_entropy_with_logits
    elif cfg.criterion == 'CE':
        return F.cross_entropy
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))


def get_label(cfg, output):
    if cfg.criterion == "BCE":
        prob = torch.sigmoid(output)
        return prob.ge(0.5).float()
    elif cfg.criterion == "CE":
        return torch.argmax(output, dim=1)


def train_epoch(summary, cfg, args, model, dataloader, optimizer,
                criterion, summary_writer):
    torch.set_grad_enabled(True)
    model.train()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)

    time_now = time.time()
    loss_sum = 0
    acc_sum = 0
    for step in range(steps):
        image, target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        output = model(image)

        acc_t = Variable(torch.Tensor(len(output)).type_as(output[0].data))
        loss_t = Variable(torch.Tensor(len(output)).type_as(output[0].data))
        for index in range(len(output)):
            label = get_label(cfg, output[index])
            acc_t[index] = (target[:, index] == label.float()).float().sum() \
                / dataloader.batch_size / len(cfg.num_classes)

            loss_t[index] = criterion(output[index], target[:, index].long())

        acc_sum += acc_t.sum()
        loss_sum += loss_t.sum()

        optimizer.zero_grad()
        loss_t.sum().backward()
        optimizer.step()

        summary['step'] += 1

        if summary['step'] % cfg.log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg.log_every
            acc_sum /= cfg.log_every

            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Loss : {:.7f}, '
                'Acc : {:.3f}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], loss_sum,
                        acc_sum, time_spent))

            summary_writer.add_scalar(
                'train/loss', loss_sum, summary['step'])
            summary_writer.add_scalar(
                'train/acc', acc_sum, summary['step'])

            loss_sum = 0
            acc_sum = 0

    summary['epoch'] += 1

    return summary


def test_epoch(summary, cfg, args, model, criterion, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)

    loss_sum = 0
    acc_sum = 0
    for step in range(steps):
        image, target = next(dataiter)

        image = image.to(device)
        target = target.to(device)
        output = model(image)
        acc_t = Variable(torch.Tensor(len(output)).type_as(output[0].data))
        loss_t = Variable(torch.Tensor(len(output)).type_as(output[0].data))
        for index in range(len(output)):
            label = get_label(cfg, output[index])
            acc_t[index] = (target[:, index] == label.float()).float().sum() \
                / dataloader.batch_size / len(cfg.num_classes)

            loss_t[index] = criterion(output[index], target[:, index].long())

        acc_sum += acc_t.sum()
        loss_sum += loss_t.sum()

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary


def run(args):
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
        if args.verbose is True:
            print(json.dumps(cfg, indent=4))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
            .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg)
    if args.verbose is True:
        from torchsummary import summary
        summary(model.to(device), (3, cfg.height, cfg.width))
    model = DataParallel(model, device_ids=device_ids).to(device).train()

    optimizer = get_optimizer(model.parameters(), cfg)
    criterion = get_criterion(cfg)

    with open(cfg.train_csv) as f:
        train_header = f.readline().strip('\n').split(',')
    with open(cfg.train_csv) as f:
        dev_header = f.readline().strip('\n').split(',')
    assert train_header == dev_header

    copyfile(cfg.train_csv, os.path.join(args.save_path, 'train.csv'))
    copyfile(cfg.dev_csv, os.path.join(args.save_path, 'dev.csv'))

    dataloader_train = DataLoader(
        ImageDataset(cfg.train_csv, cfg, mode='train'),
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=True)
    dataloader_dev = DataLoader(
        ImageDataset(cfg.dev_csv, cfg, mode='dev'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    summary_train = {'epoch': 0, 'step': 0}
    summary_dev = {'loss': float('inf'), 'acc': 0.0}
    summary_writer = SummaryWriter(args.save_path)
    acc_dev_best = 0.0
    loss_dev_best = 9999999
    fused_best = 0
    epoch_start = 0
    best_count = 0

    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'train.ckpt')
        ckpt = torch.load(ckpt_path)
        model.module.load_state_dict(ckpt['state_dict'])
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        acc_dev_best = ckpt['acc_dev_best']
        loss_dev_best = ckpt['loss_dev_best']
        epoch_start = ckpt['epoch']

    for epoch in range(epoch_start, cfg.epoch):
        lr = lr_schedule(cfg.lr, cfg.lr_factor, summary_train['epoch'],
                         cfg.lr_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        summary_train = train_epoch(summary_train, cfg, args, model,
                                    dataloader_train, optimizer, criterion,
                                    summary_writer)

        time_now = time.time()
        summary_dev = test_epoch(summary_dev, cfg, args, model, criterion,
                                 dataloader_dev)
        time_spent = time.time() - time_now

        logging.info(
            '{}, Dev, Epoch : {}, Step : {}, Loss : {:.7f}, Acc : {:.3f} '
            'Run Time : {:.2f} sec'
            .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['epoch'], summary_train['step'],
                    summary_dev['loss'], summary_dev['acc'],
                    time_spent))

        summary_writer.add_scalar('dev/loss',
                                  summary_dev['loss'],
                                  summary_train['step'])
        summary_writer.add_scalar('dev/acc',
                                  summary_dev['acc'],
                                  summary_train['step'])

        if cfg.best_target == "acc":
            if summary_dev['acc'] > acc_dev_best:
                best_count = best_count + 1
                acc_dev_best = summary_dev['acc']
                loss_dev_best = summary_dev['loss']
                torch.save({'epoch': summary_train['epoch'],
                            'step': summary_train['step'],
                            'acc_dev_best': acc_dev_best,
                            'loss_dev_best': loss_dev_best,
                            'state_dict': model.module.state_dict()},
                           os.path.join(args.save_path, 'acc_best_' + str(best_count) + '.ckpt')
                           )

                logging.info(
                    '{}, Best, Epoch : {}, Step : {}, Loss : {:.7f}, Acc : {:.3f}'
                        .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                summary_train['epoch'], summary_train['step'],
                                summary_dev['loss'],
                                summary_dev['acc']))
        elif cfg.best_target == "loss":
            if summary_dev['loss'] < loss_dev_best:
                best_count = best_count + 1
                acc_dev_best = summary_dev['acc']
                loss_dev_best = summary_dev['loss']
                torch.save({'epoch': summary_train['epoch'],
                            'step': summary_train['step'],
                            'acc_dev_best': acc_dev_best,
                            'loss_dev_best': loss_dev_best,
                            'state_dict': model.module.state_dict()},
                           os.path.join(args.save_path, 'loss_best_' + str(best_count) + '.ckpt')
                           )

                logging.info(
                    '{}, Best, Epoch : {}, Step : {}, Loss : {:.7f}, Acc : {:.3f}'
                        .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                summary_train['epoch'], summary_train['step'],
                                summary_dev['loss'],
                                summary_dev['acc']))
        elif cfg.best_target == "fused":
            if summary_dev['acc'] - summary_dev['loss'] > fused_best:
                best_count = best_count + 1
                acc_dev_best = summary_dev['acc']
                fused_best = summary_dev['acc'] - summary_dev['loss']
                loss_dev_best = summary_dev['loss']
                torch.save({'epoch': summary_train['epoch'],
                            'step': summary_train['step'],
                            'acc_dev_best': acc_dev_best,
                            'loss_dev_best': loss_dev_best,
                            'state_dict': model.module.state_dict()},
                           os.path.join(args.save_path, 'fused_best_' + str(best_count) + '.ckpt')
                           )

                logging.info(
                    '{}, Best, Epoch : {}, Step : {}, Loss : {:.7f}, Acc : {:.3f}'
                        .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                summary_train['epoch'], summary_train['step'],
                                summary_dev['loss'],
                                summary_dev['acc']))
        else:
            raise Exception(
                'Unknown best_target type : {}'.format(cfg.best_target)
            )

        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'acc_dev_best': acc_dev_best,
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))

    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)


if __name__ == '__main__':
    main()
