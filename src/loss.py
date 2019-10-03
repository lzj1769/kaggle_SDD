import torch
import torch.nn as nn
import numpy as np
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        probability = torch.sigmoid(input)
        batch_size = target.shape[0]
        channel_num = target.shape[1]
        dice = 0.0
        for i in range(batch_size):
            for j in range(channel_num):
                channel_loss = self.dice_loss_single_channel(probability[i, j, :, :], target[i, j, :, :])
                dice += channel_loss / (batch_size * channel_num)

        return dice

    def dice_loss_single_channel(self, input, target):
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()

        return 1 - ((2. * intersection + self.smooth) /
                    (input_flat.sum() + target_flat.sum() + self.smooth))


class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def forward(self, input, target):
        batch_size = target.shape[0]
        channel_num = target.shape[1]
        loss = 0.0
        for i in range(batch_size):
            for j in range(channel_num):
                channel_loss = self.lovasz_loss_single_channel(input[i, j, :, :], target[i, j, :, :])
                loss += channel_loss / (batch_size * channel_num)

        return loss

    def lovasz_loss_single_channel(self, input, target):
        input = input.view(-1)
        target = target.view(-1)

        signs = 2. * target.float() - 1.
        errors = (1. - input * Variable(signs))

        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = target[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))

        return loss

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


class DiceBCELoss(nn.Module):
    def __init__(self, alpha=0.9, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.alpha = alpha
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        dice_loss = self.dice(input, target)
        loss = self.alpha * bce_loss + (1 - self.alpha) * dice_loss

        return loss, bce_loss, dice_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets,
                                                      reduction='none')
        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        loss = (1 - pt) ** self.gamma * bce_loss
        return loss.mean()
