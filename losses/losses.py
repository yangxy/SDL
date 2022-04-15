import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


"""
LapStyle loss
"""
def calc_emd_loss(pred, target):
    """calculate emd loss.
    Args:
        pred (Tensor): of shape (N, C, H, W). Predicted tensor.
        target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    """
    b, _, h, w = pred.shape
    pred = pred.reshape([b, -1, w * h])
    pred_norm = torch.sqrt((pred**2).sum(1).reshape([b, -1, 1]))
    #pred = pred.transpose([0, 2, 1])
    pred = pred.permute(0, 2, 1)
    target_t = target.reshape([b, -1, w * h])
    target_norm = torch.sqrt((target**2).sum(1).reshape([b, 1, -1]))
    similarity = torch.bmm(pred, target_t) / pred_norm / target_norm
    dist = 1. - similarity
    return dist

def calc_mean_std(feat, eps=1e-5):
    """calculate mean and standard deviation.
    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).
        eps (float): Default: 1e-5.
    Return:
        mean and std of feat
        shape: [N, C, 1, 1]
    """
    size = feat.shape
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape([N, C, -1])
    feat_var = torch.var(feat_var, axis=2) + eps
    feat_std = torch.sqrt(feat_var)
    feat_std = feat_std.reshape([N, C, 1, 1])
    feat_mean = feat.reshape([N, C, -1])
    feat_mean = torch.mean(feat_mean, axis=2)
    feat_mean = feat_mean.reshape([N, C, 1, 1])
    return feat_mean, feat_std


def mean_variance_norm(feat):
    """mean_variance_norm.
    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).
    Return:
        Normalized feat with shape (N, C, H, W)
    """
    size = feat.shape
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


@LOSS_REGISTRY.register()
class ContentStyleReltLoss(nn.Module):
    """Calc Content Relt Loss.
    """
    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 content_weight=1.0,
                 style_weight=1.0,
                 criterion='l1'):
        super(ContentStyleReltLoss, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

    def forward(self, x, content, style):
        """Forward Function.
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        x_features = self.vgg(x)
        c_features = self.vgg(content.detach())
        s_features = self.vgg(style.detach())

        style_loss, content_loss = 0, 0
        for k in x_features.keys():
            dM = 1.
            Mx = calc_emd_loss(x_features[k], x_features[k])
            Mx = Mx / Mx.sum(1, keepdim=True)
            My = calc_emd_loss(c_features[k], c_features[k])
            My = My / My.sum(1, keepdim=True)
            content_loss = content_loss + torch.abs(dM * (Mx - My)).mean() * x_features[k].shape[2] * x_features[k].shape[3]

            CX_M = calc_emd_loss(x_features[k], s_features[k])
            #m1 = CX_M.min(2)
            #m2 = CX_M.min(1)
            #m = torch.cat([m1.mean(), m2.mean()])
            #style_loss = style_loss + torch.max(m)
            m1 = CX_M.min(2)[0].data
            m2 = CX_M.min(1)[0].data
            style_loss = style_loss + torch.max(m1.mean(), m2.mean())

        return self.content_weight * content_loss, self.style_weight * style_loss


@LOSS_REGISTRY.register()
class ContentStyleLoss(nn.Module):
    """Calc Content Loss.
    """
    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 content_weight=1.0,
                 style_weight=1.0,
                 criterion='l1'):
        super(ContentStyleLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

    def forward(self, x, content, style, norm=False):
        """Forward Function.
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            norm(Bool): whether use mean_variance_norm for pred and target
        """
        x_features = self.vgg(x)
        c_features = self.vgg(content.detach())
        s_features = self.vgg(style.detach())

        style_loss, content_loss = 0, 0
        for k in x_features.keys():
            if (norm == False):
                content_loss = content_loss + self.mse_loss(x_features[k], c_features[k])
            else:
                content_loss = content_loss + self.mse_loss(mean_variance_norm(x_features[k]),
                                     mean_variance_norm(c_features[k]))

            pred_mean, pred_std = calc_mean_std(x_features[k])
            target_mean, target_std = calc_mean_std(s_features[k])
            style_loss = style_loss + self.mse_loss(pred_mean, target_mean) + self.mse_loss(pred_std, target_std)

        return self.content_weight * content_loss, self.style_weight * style_loss





