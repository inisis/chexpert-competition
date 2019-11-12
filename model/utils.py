import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from model.pooling import LogSumExpPool, ExpPool, LinearPool

def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == 'BatchNorm':
        return nn.BatchNorm2d(num_features, eps=eps)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(num_groups, num_features, eps=eps)
    elif norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(num_features, eps=eps,
                                 affine=True, track_running_stats=True)
    else:
        raise Exception('Unknown Norm Function : {}'.format(norm_type))


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


def get_pooling(cfg, index, training):
    if cfg.pooling[index] == 'AVG':
        return nn.AdaptiveAvgPool2d((1, 1))
    elif cfg.pooling[index] == 'LSE':
        return LogSumExpPool(cfg.lse_gamma)
    elif cfg.pooling[index] == 'PROB':
        return ProbPool()
    elif cfg.pooling[index] == 'EXP':
        return ExpPool()
    elif cfg.pooling[index] == 'LINEAR':
        return LinearPool()
    else:
        raise Exception('Unknown Pooling: {}'.format(cfg.pooling[index]))

def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
    """
    Learning rate schedule with respect to epoch
    lr: float, initial learning rate
    lr_factor: float, decreasing factor every epoch_lr
    epoch_now: int, the current epoch
    lr_epochs: list of int, decreasing every epoch in lr_epochs
    return: lr, float, scheduled learning rate.
    """
    count = 0
    for epoch in lr_epochs:
        if epoch_now >= epoch:
            count += 1
            continue

        break

    return lr * np.power(lr_factor, count)

def get_loss(output, target, index, device, cfg):
    if cfg.criterion == 'BCE' and not cfg.loss_batch_weight:
        assert len(cfg.pos_weight) == cfg.num_tasks
        pos_weight = torch.from_numpy(
            np.array(cfg.pos_weight,
                     dtype=np.float32)).to(device).type_as(target)
        loss = F.binary_cross_entropy_with_logits(output[:, index],
                                                  target[:, index],
                                                  pos_weight=pos_weight[index])
        label = torch.sigmoid(output[:, index]).ge(0.5).float()
        acc = (target[:, index] == label).float().sum() / len(label)
    elif cfg.criterion == 'BCE' and cfg.loss_batch_weight:
        label = torch.sigmoid(output[:, index]).ge(0.5).float()
        p_N = target.sum(dim=0)
        n_N = len(label) - p_N
        loss = 0
        for i in range(len(label)):
            loss_i = F.binary_cross_entropy_with_logits(output[i, index],
                                                        target[i, index])
            if target[i, index] == 1:
                loss += loss_i * (n_N[index]/len(label))
            elif target[i, index] == 0:
                loss += loss_i * (p_N[index]/len(label))
        loss = loss / len(label)
        acc = (target[:, index] == label).float().sum() / len(label)
    elif cfg.criterion == 'FL':
        input = output[:, index]
        max_val = (-input).clamp(min=0)
        loss = input - input * target[:, index] + max_val +((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target[:, index] * 2 - 1))
        loss = (invprobs * cfg.gamma).exp() * loss * 0.25
        loss = loss.mean()
        label = torch.sigmoid(output[:, index].view(-1)).ge(0.5).float()
        acc = (target[:, index] == label).float().sum() / len(label)
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return (loss, acc)


def get_pred(output, index, cfg):
    if cfg.criterion == 'BCE' or cfg.criterion == 'FL':
        pred = torch.sigmoid(output[:, index]).cpu().detach().numpy()
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return pred


def read_csv(csv_path, dev=False):
    image_paths = []
    probs = []

    if dev:
        dict_ = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                    {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'},]
        with open(csv_path) as f:
            header = f.readline().strip('\n').split(',')
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_paths.append(fields[0])
                for index, value in enumerate(fields[1:]):
                    if index == 1 or index == 3:
                        labels.append(dict_[1].get(value))
                    elif index == 0 or index == 2 or index == 4:
                        labels.append(dict_[0].get(value))
                probs.append(list(map(int, labels)))
        probs = np.array(probs)

        return (image_paths, probs, header)

    with open(csv_path) as f:
        header = f.readline().strip('\n').split(',')
        for line in f:
            fields = line.strip('\n').split(',')
            image_paths.append(fields[0])
            probs.append(list(map(float, fields[1:])))
    probs = np.array(probs)

    return (image_paths, probs, header)

def from_list(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x
