from torch import nn

from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.resnet import (resnet18, resnet34, resnet50, resnet101,
                                   resnet152)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)


BACKBONES = {'vgg19': vgg19,
             'vgg19_bn': vgg19_bn,
             'resnet18': resnet18,
             'resnet34': resnet34,
             'resnet50': resnet50,
             'resnet101': resnet101,
             'resnet152': resnet152,
             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             'inception_v3': inception_v3}


BACKBONES_TYPES = {'vgg19': 'vgg',
                   'vgg19_bn': 'vgg',
                   'resnet18': 'resnet',
                   'resnet34': 'resnet',
                   'resnet50': 'resnet',
                   'resnet101': 'resnet',
                   'resnet152': 'resnet',
                   'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet',
                   'inception_v3': 'inception'}


class Classifier(nn.Module):

    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONES[cfg.backbone](cfg.pretrained)
        self._init_classifier()

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(self, "fc_" + str(index), nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, num_class),
                ))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'resnet':
                setattr(self, "fc_" + str(index),
                        nn.Linear(512 * self.backbone.block.expansion,
                                  num_class))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(self, "fc_" + str(index),
                        nn.Linear(self.backbone.num_features, num_class))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(self, "fc_" + str(index),
                        nn.Linear(2048, num_class))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Linear):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        x = self.backbone.forward(x)
        outs = list()
        for index, num_task in enumerate(self.cfg.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            out = classifier(x)
            outs.append(out)
        return outs
