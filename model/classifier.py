import torch
from torch import nn

import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.resnet import (resnet18, resnet34, resnet50, resnet101,
                                   resnet152, resnext50_32x4d,
                                   resnext101_32x8d)
from model.backbone.resnetfpn5 import (resnet18fpn5, resnet34fpn5,
                                       resnet50fpn5, resnet101fpn5,
                                       resnet152fpn5, resnext50_32x4dfpn5,
                                       resnext101_32x8dfpn5)
from model.backbone.resnetfpn6 import (resnet18fpn6, resnet34fpn6,
                                       resnet50fpn6, resnet101fpn6,
                                       resnet152fpn6, resnext50_32x4dfpn6,
                                       resnext101_32x8dfpn6)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)
from model.backbone.senet import senet154
from model.attention_map import AttentionMap


from model.utils import get_pooling


BACKBONES = {'vgg19': vgg19,
             'vgg19_bn': vgg19_bn,
             'resnet18': resnet18,
             'resnet34': resnet34,
             'resnet50': resnet50,
             'resnet101': resnet101,
             'resnet152': resnet152,
             'resnext50_32x4d': resnext50_32x4d,
             'resnext101_32x8d': resnext101_32x8d,
             'resnet18fpn5': resnet18fpn5,
             'resnet34fpn5': resnet34fpn5,
             'resnet50fpn5': resnet50fpn5,
             'resnet101fpn5': resnet101fpn5,
             'resnet152fpn5': resnet152fpn5,
             'resnext50_32x4dfpn5': resnext50_32x4dfpn5,
             'resnext101_32x8dfpn5': resnext101_32x8dfpn5,
             'resnet18fpn6': resnet18fpn6,
             'resnet34fpn6': resnet34fpn6,
             'resnet50fpn6': resnet50fpn6,
             'resnet101fpn6': resnet101fpn6,
             'resnet152fpn6': resnet152fpn6,
             'resnext50_32x4dfpn6': resnext50_32x4dfpn6,
             'resnext101_32x8dfpn6': resnext101_32x8dfpn6,
             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             'inception_v3': inception_v3}
             

class Classifier(nn.Module):

    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONES[cfg.backbone](cfg)
        self.global_pooling = []
        self.attention_map = AttentionMap(self.cfg,
                                          self.backbone.num_features)
        
        for index in range(self.cfg.num_tasks):
            expand = 1
            if self.cfg.pooling[index] == 'AVG_MAX':
                expand = 2
            setattr(self, "fc_" + str(index),
                    nn.Conv2d(self.backbone.num_features * expand, 1, kernel_size=1,
                              stride=1, padding=0, bias=True))
            self.global_pooling.append(get_pooling(self.cfg, index, self.training))
            classifier = getattr(self, "fc_" + str(index))
            classifier.weight.data.normal_(0, 0.01)
            classifier.bias.data.zero_()
            if self.cfg.fc_bn:
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(self.backbone.num_features * expand))


    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        feat_map = self.backbone(x)
        if self.cfg.attention_map != "None":
            feat_map = self.attention_map(feat_map)
        (N, C, H, W) = tuple(feat_map.shape)
        logits = torch.zeros(N, self.cfg.num_tasks).to(feat_map.device)
        logit_maps = torch.zeros(N, self.cfg.num_tasks,
                                 H, W).to(feat_map.device)

        for index, num_class in enumerate(self.cfg.num_classes):
            
            classifier = getattr(self, "fc_" + str(index))
            logit_map = None
            if not (self.cfg.pooling[index] == 'AVG_MAX'):
                logit_map = classifier(feat_map)
                logit_maps[:, index: index+1, ...] = logit_map
            feat = self.global_pooling[index](feat_map)
            if self.cfg.fc_bn:
                bn = getattr(self, "bn_" + str(index))
                feat = bn(feat)
            feat = F.dropout(feat, training=self.training)

            logit = classifier(feat)
            logit = logit.squeeze(-1).squeeze(-1)
            logits[:, index: index+1] = logit

        return (logits, logit_maps)
