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


class SymDiceBCELoss(nn.Module):
    def __init__(self, alpha=0.9, smooth=1):
        super(SymDiceBCELoss, self).__init__()
        self.alpha = alpha
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        dice_loss_positive = self.dice(input, target)
        dice_loss_negative = self.dice(-input, 1-target)
        dice_loss = (dice_loss_positive + dice_loss_negative) / 2
        loss = self.alpha * bce_loss + (1 - self.alpha) * dice_loss

        return loss, bce_loss, dice_loss
