import os
import logging
import json
import time
import subprocess
from shutil import copyfile
from sklearn import metrics

from easydict import EasyDict as edict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import Augmentor
import torchvision
from tensorboardX import SummaryWriter

from data.dataset import CSVDataset  # noqa
from model.classifier import Classifier  # noqa
from model.utils import get_optimizer, lr_schedule, get_loss, get_pred, read_csv, from_list  # noqa

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

MAX_DIRECTORY_SIZE = 10 * 1024
MAX_WORKERS_PER_GPU = 8


class Trainer(object):

    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

    def setup(self):
        # create cfg
        with open(self.args.cfg_path) as f:
            cfg = edict(json.load(f))
            if self.args.verbose is True:
                print(json.dumps(cfg, indent=4))

        # create save path
        if not os.path.exists(self.args.save_path):
            os.mkdir(self.args.save_path)

        # create log.txt
        if self.args.logtofile is True:
            logging.basicConfig(filename=self.args.save_path + '/log.txt',
                                filemode="w", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)

        # dump cfg.json
        with open(os.path.join(self.args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

        # check device
        device_ids = list(map(int, self.args.device_ids.split(',')))
        assert len(device_ids) * MAX_WORKERS_PER_GPU >= self.args.num_workers
        num_devices = torch.cuda.device_count()
        if num_devices < len(device_ids):
            raise Exception(
                '#available gpu : {} < --device_ids : {}'
                .format(num_devices, len(device_ids)))

        # check header consistent
        with open(cfg.train_csv) as f:
            train_header = f.readline().strip('\n').split(',')[1:]
        with open(cfg.dev_csv) as f:
            dev_header = f.readline().strip('\n').split(',')[1:]
        assert train_header == dev_header

        # check repo size
        src_folder = os.path.dirname(os.path.abspath(__file__)) + '/../'
        dst_folder = os.path.join(self.args.save_path, 'classification')
        rc, size = subprocess.getstatusoutput('du --max-depth=0 %s | cut -f1'
                                              % src_folder)
        if rc != 0:
            raise Exception('Copy folder error : {}'.format(rc))
        if int(size) > MAX_DIRECTORY_SIZE:
            raise Exception('Repo size too large : {}'.format(int(size)))
        rc, err_msg = subprocess.getstatusoutput('cp -R %s %s' % (src_folder,
                                                 dst_folder))
        if rc != 0:
            raise Exception('Copy folder error : {}'.format(err_msg))

        # backup repo
        copyfile(cfg.train_csv, os.path.join(self.args.save_path, 'train.csv'))
        copyfile(cfg.dev_csv, os.path.join(self.args.save_path, 'dev.csv'))

        # init other members
        self.cfg = cfg

        self.device = torch.device('cuda:{}'.format(device_ids[0]))

        model = Classifier(cfg)
        if self.args.verbose is True:
            from torchsummary import summary
            if self.cfg.fix_ratio:
                h, w = self.cfg.long_side, self.cfg.long_side
            else:
                h, w = self.cfg.height, self.cfg.width
            summary(model.to(self.device), (3, h, w))
        self.model = DataParallel(
            model, device_ids=device_ids
        ).to(self.device).train()

        self.optimizer = get_optimizer(model.parameters(), cfg)
        self.p = Augmentor.Pipeline()
        self.p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)
        
        self.transforms = torchvision.transforms.Compose([
                         self.p.torch_transform(),
                         from_list
                     ])
        self.dataloader_train = DataLoader(
            CSVDataset(self.cfg.train_csv, self.cfg, mode='train', transform=self.transforms),
            batch_size=self.cfg.train_batch_size,
            num_workers=self.args.num_workers,
            drop_last=False, shuffle=True)
        self.dataiter_train = iter(self.dataloader_train)

        if self.cfg.best_target == 'auc':
            mode = 'test'
        else:
            mode = 'dev'

        self.dataloader_dev = DataLoader(
            CSVDataset(self.cfg.dev_csv, self.cfg, mode=mode),
            batch_size=self.cfg.dev_batch_size,
            num_workers=self.args.num_workers,
            drop_last=False, shuffle=False)
        self.dataiter_dev = iter(self.dataloader_dev)

        self.summary_writer = SummaryWriter(self.args.save_path)

        self.summary = {'step': 0, 'log_step': 0, 'epoch': 1,
                        'loss_sum_train': np.zeros(self.cfg.num_tasks),
                        'acc_sum_train': np.zeros(self.cfg.num_tasks),
                        'acc_dev': 0.0,
                        'auc_dev': np.zeros(self.cfg.num_tasks),
                        'loss_dev': float('inf'), 'acc_dev_best': 0.0,
                        'loss_dev_best': float('inf'), 'fused_dev_best': 0.0,
                        'auc_dev_best': np.zeros(self.cfg.num_tasks),
                        'auc_dev_best_each': np.zeros(self.cfg.num_tasks)}

        lr = lr_schedule(self.cfg.lr, self.cfg.lr_factor, self.summary['epoch'],
                         self.cfg.lr_epochs)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # global time, loss, acc
        self.time_now = time.time()
        self.label_header = self.dataloader_train.dataset._label_header
        
        torch.set_grad_enabled(True)


    def log_init(self):
        self.summary['loss_sum_train'] = np.zeros(self.cfg.num_tasks)
        self.summary['acc_sum_train'] = np.zeros(self.cfg.num_tasks)
        self.summary['log_step'] = 0

    def train_step(self):
        try:
            image, target = next(self.dataiter_train)
        except StopIteration:
            self.summary['epoch'] += 1
            # may wrap into a funcion
            lr = lr_schedule(self.cfg.lr, self.cfg.lr_factor, self.summary['epoch'],
                             self.cfg.lr_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.dataiter_train = iter(self.dataloader_train)
            image, target = next(self.dataiter_train)

        image = image.to(self.device)
        target = target.to(self.device)
        output, logit_map = self.model(image)

        # different number of tasks
        loss = 0
        for t in range(self.cfg.num_tasks):
            loss_t, acc_t = get_loss(output, target, t, self.device, self.cfg)
            loss_t *= self.cfg.loss_weight[t]
            loss += loss_t
            self.summary['loss_sum_train'][t] += loss_t.item()
            self.summary['acc_sum_train'][t] += acc_t.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.summary['step'] += 1
        self.summary['log_step'] += 1


    def dev_epoch(self):
        self.time_now = time.time()
        torch.set_grad_enabled(False)
        self.model.eval()
        steps = len(self.dataloader_dev)
        dataiter = iter(self.dataloader_dev)
        
        if self.cfg.best_target == 'auc':
            with open(self.cfg.train_csv) as f:
                test_header = f.readline().strip('\n').split(',')

            with open(self.args.save_path + '/predict.csv', 'w') as f:
                f.write(','.join(test_header) + '\n')
                for step in range(steps):
                    image, path = next(dataiter)
                    image = image.to(self.device)
                    output, logit_map = self.model(image)
                    batch_size = len(path)
                    pred = np.zeros((self.cfg.num_tasks, batch_size))
                    for t in range(self.cfg.num_tasks):
                        pred[t] = get_pred(output, t, self.cfg)
                    for i in range(batch_size):
                        probs = ','.join(map(lambda x: '{:.5f}'.format(x), pred[:, i]))
                        result = path[i] + ',' + probs
                        f.write(result + '\n')
            
            images_pred, probs_pred, header_pred = read_csv(self.args.save_path + '/predict.csv')
            images_true, probs_true, header_true = read_csv(self.cfg.dev_csv, True)

            assert header_pred == header_true
            assert images_pred == images_true

            num_labels = len(header_true) - 1
            for i in range(num_labels):
                label = header_true[i+1]
                y_pred = probs_pred[:, i]
                y_true = probs_true[:, i]
                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                self.summary['auc_dev'][i] = auc

        else:
            loss_sum = np.zeros(self.cfg.num_tasks)
            acc_sum = np.zeros(self.cfg.num_tasks)
            for step in range(steps):
                image, target = next(dataiter)
                image = image.to(self.device)
                target = target.to(self.device)
                output, logit_map = self.model(image)
                # different number of tasks
                for t in range(self.cfg.num_tasks):
                    loss_t, acc_t = get_loss(output, target, t, self.device, self.cfg)
                    loss_sum[t] += loss_t.item()
                    acc_sum[t] += acc_t.item()
            self.summary['loss_dev'] = loss_sum / steps
            self.summary['acc_dev'] = acc_sum / steps

        torch.set_grad_enabled(True)
        self.model.train()


    def logging(self, mode='Train'):
        time_spent = time.time() - self.time_now
        self.time_now = time.time()
        
        if mode == 'Train':
            loss_train = self.summary['loss_sum_train'] / self.summary['log_step']
            acc_train = self.summary['acc_sum_train'] / self.summary['log_step']
            loss_train_str = ' '.join(map(lambda x: '{:.5f}'.format(x), loss_train))
            acc_train_str = ' '.join(map(lambda x: '{:.3f}'.format(x), acc_train))

            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Loss : {}, '
                'Acc : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        self.summary['epoch'], self.summary['step'], loss_train_str,
                        acc_train_str, time_spent))
        elif mode == 'Dev' and self.cfg.best_target != 'auc':
            loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                    self.summary['loss_dev']))
            acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   self.summary['acc_dev']))

            logging.info(
                '{}, Dev, Epoch : {}, Step : {}, Loss : {}, Acc : {} '
                'Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        self.summary['epoch'], self.summary['step'],
                        loss_dev_str, acc_dev_str, time_spent))
        elif mode == 'Dev' and self.cfg.best_target == 'auc':
            auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                       self.summary['auc_dev']))

            logging.info(
                '{}, Dev, Epoch : {}, Step : {}, Auc : {} '
                'Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        self.summary['epoch'], self.summary['step'],
                        auc_dev_str, time_spent))


    def write_summary(self, mode='Train'):
        if mode == 'Train':
            for t in range(self.cfg.num_tasks):
                self.summary_writer.add_scalar(
                    'Train/loss_{}'.format(self.label_header[t]),
                    self.summary['loss_sum_train'][t] / self.summary['log_step'],
                    self.summary['step'])
                self.summary_writer.add_scalar(
                    'Train/acc_{}'.format(self.label_header[t]),
                    self.summary['acc_sum_train'][t] / self.summary['log_step'],
                    self.summary['step'])
        elif mode == 'Dev' and self.cfg.best_target != 'auc':
            for t in range(self.cfg.num_tasks):
                self.summary_writer.add_scalar(
                    'Dev/loss_{}'.format(self.label_header[t]), self.summary['loss_dev'][t],
                    self.summary['step'])
                self.summary_writer.add_scalar(
                    'Dev/acc_{}'.format(self.label_header[t]), self.summary['acc_dev'][t],
                    self.summary['step'])
        elif mode == 'Dev' and self.cfg.best_target == 'auc':
            for t in range(self.cfg.num_tasks):
                self.summary_writer.add_scalar(
                    'Dev/auc_{}'.format(self.label_header[t]), self.summary['auc_dev'][t],
                    self.summary['step'])


    def save_model(self, mode='Train'):
        if mode == 'Train':    
            torch.save({'epoch': self.summary['epoch'],
                        'step': self.summary['step'],
                        'acc_dev_best': self.summary['acc_dev_best'],
                        'loss_dev_best': self.summary['loss_dev_best'],
                        'state_dict': self.model.module.state_dict()},
                       os.path.join(self.args.save_path, 'train.ckpt'))
        elif mode == 'Dev' and self.cfg.best_target != 'auc':
            save_best = False
            if self.summary['acc_dev'].mean() > self.summary['acc_dev_best']:
                self.summary['acc_dev_best'] = self.summary['acc_dev'].mean()
                if self.cfg.best_target == 'acc':
                    save_best = True
            if self.summary['loss_dev'].mean() < self.summary['loss_dev_best']:
                self.summary['loss_dev_best'] = self.summary['loss_dev'].mean()
                if self.cfg.best_target == 'loss':
                    save_best = True
            if self.summary['acc_dev'].mean() - self.summary['loss_dev'].mean() > \
                    self.summary['fused_dev_best']:
                self.summary['fused_dev_best'] = self.summary['acc_dev'].mean() - \
                    self.summary['loss_dev'].mean()
                if self.cfg.best_target == 'fused':
                    save_best = True

            if save_best:
                torch.save(
                    {'epoch': self.summary['epoch'],
                     'step': self.summary['step'],
                     'acc_dev_best': self.summary['acc_dev_best'],
                     'loss_dev_best': self.summary['loss_dev_best'],
                     'state_dict': self.model.module.state_dict()},
                    os.path.join(self.args.save_path, 'best.ckpt')
                )
                loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                        self.summary['loss_dev']))
                acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       self.summary['acc_dev']))

                logging.info(
                    '{}, Best, Epoch : {}, Step : {}, Loss : {}, Acc : {}'
                    .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                            self.summary['epoch'], self.summary['step'],
                            loss_dev_str, acc_dev_str))

        elif mode =='Dev' and self.cfg.best_target == 'auc':
            if self.summary['auc_dev_best'].mean() < self.summary['auc_dev'].mean():
                self.summary['auc_dev_best'] = self.summary['auc_dev'].copy()
                torch.save(
                    {'epoch': self.summary['epoch'],
                     'step': self.summary['step'],
                     'auc_dev_best': self.summary['auc_dev_best'],
                     'state_dict': self.model.module.state_dict()},
                    os.path.join(self.args.save_path, 'best.ckpt')
                )
                auc_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                        self.summary['auc_dev_best']))
                logging.info(
                    '{}, Best, Epoch : {}, Step : {}, Auc : {}'
                    .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                            self.summary['epoch'], self.summary['step'],
                            auc_dev_str))
            for t in range(self.cfg.num_tasks):
                if(self.summary['auc_dev_best_each'][t] <= self.summary['auc_dev'][t]):
                    self.summary['auc_dev_best_each'][t] = self.summary['auc_dev'][t].copy()
                    model_name = self.label_header[t] + '_best.ckpt'
                    torch.save(
                        {'epoch': self.summary['epoch'],
                         'step': self.summary['step'],
                         'auc_dev_best': self.summary['auc_dev_best_each'],
                         'state_dict': self.model.module.state_dict()},
                        os.path.join(self.args.save_path, model_name)
                    )
                    logging.info(
                        '{}, Best, Epoch : {}, Step : {}, Label: {}, Auc : {}'
                        .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                self.summary['epoch'], self.summary['step'],
                                self.label_header[t],
                                str(self.summary['auc_dev_best_each'][t])))


              
    def close(self):
         self.summary_writer.close()
