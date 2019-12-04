import torch
from torch import nn


class AvgMaxPool(nn.Module):
    def __init__(self):
        super(AvgMaxPool, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, feat_map):
        a = self.avgpool(feat_map)
        b = self.maxpool(feat_map)
        return torch.cat((a, b), 1)


class ProbPool(nn.Module):

    def __init__(self):
        super(ProbPool, self).__init__()

    def forward(self, feat_map, logit_map):
        prob_map = torch.sigmoid(logit_map)
        weight_map = prob_map / prob_map.sum(dim=2, keepdim=True)\
            .sum(dim=3, keepdim=True)
        feat = (feat_map * weight_map).sum(dim=2, keepdim=True)\
            .sum(dim=3, keepdim=True)

        return feat


class LogSumExpPool(nn.Module):

    def __init__(self, gamma):
        super(LogSumExpPool, self).__init__()
        self.gamma = gamma

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        (N, C, H, W) = feat_map.shape

        # (N, C, 1, 1) m
        m, _ = torch.max(
            feat_map, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)

        # (N, C, H, W) value0
        value0 = feat_map - m
        area = 1.0 / (H * W)
        g = self.gamma

        # TODO: split dim=(-1, -2) for onnx.export
        return m + 1 / g * torch.log(area * torch.sum(torch.exp(g * value0),
                                     dim=(-1, -2), keepdim=True))


class ExpPool(nn.Module):

    def __init__(self):
        super(ExpPool, self).__init__()

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """

        EPSILON = 1e-7
        (N, C, H, W) = feat_map.shape
        m, _ = torch.max(
            feat_map, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)

        # caculate the sum of exp(xi)
        # TODO: split dim=(-1, -2) for onnx.export
        sum_exp = torch.sum(torch.exp(feat_map - m),
                            dim=(-1, -2), keepdim=True)

        # prevent from dividing by zero
        sum_exp += EPSILON

        # caculate softmax in shape of (H,W)
        exp_weight = torch.exp(feat_map - m) / sum_exp
        weighted_value = feat_map * exp_weight

        # TODO: split dim=(-1, -2) for onnx.export
        return torch.sum(weighted_value, dim=(-1, -2), keepdim=True)


class LinearPool(nn.Module):

    def __init__(self):
        super(LinearPool, self).__init__()

    def forward(self, feat_map):
        """
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        EPSILON = 1e-7
        (N, C, H, W) = feat_map.shape

        # sum feat_map's last two dimention into a scalar
        # so the shape of sum_input is (N,C,1,1)
        # TODO: split dim=(-1, -2) for onnx.export
        sum_input = torch.sum(feat_map, dim=(-1, -2), keepdim=True)

        # prevent from dividing by zero
        sum_input += EPSILON

        # caculate softmax in shape of (H,W)
        linear_weight = feat_map / sum_input
        weighted_value = feat_map * linear_weight

        # TODO: split dim=(-1, -2) for onnx.export
        return torch.sum(weighted_value, dim=(-1, -2), keepdim=True)

