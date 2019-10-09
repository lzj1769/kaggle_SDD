import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss


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


class LovaseBCELoss(nn.Module):
    def __init__(self, alpha=0.9):
        super(LovaseBCELoss, self).__init__()
        self.alpha = alpha
        self.bce = BCEWithLogitsLoss()
        self.lovase = lovasz_hinge

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        lovase_loss = self.lovase(input, target)
        loss = self.alpha * bce_loss + (1 - self.alpha) * lovase_loss

        return loss, bce_loss, lovase_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()