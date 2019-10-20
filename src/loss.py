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


class DiceLossWeight(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLossWeight, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        probability = torch.sigmoid(input)
        batch_size = target.shape[0]
        channel_num = target.shape[1]
        dice1, dice2, dice3, dice4 = 0.0, 0.0, 0.0, 0.0

        for i in range(batch_size):
            dice1 += self.dice_loss_single_channel(probability[i, 0, :, :], target[i, 0, :, :])
            dice2 += self.dice_loss_single_channel(probability[i, 1, :, :], target[i, 1, :, :])
            dice3 += self.dice_loss_single_channel(probability[i, 2, :, :], target[i, 2, :, :])
            dice4 += self.dice_loss_single_channel(probability[i, 3, :, :], target[i, 3, :, :])

        dice1 /= batch_size
        dice2 /= batch_size
        dice3 /= batch_size
        dice4 /= batch_size

        return dice1, dice2, dice3, dice3

    def dice_loss_single_channel(self, input, target):
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()

        return 1 - ((2. * intersection + self.smooth) /
                    (input_flat.sum() + target_flat.sum() + self.smooth))


class DiceBCELossWeight(nn.Module):
    def __init__(self, alpha=0.9, smooth=1):
        super(DiceBCELossWeight, self).__init__()
        self.alpha = alpha
        self.bce = BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLossWeight(smooth)

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        bce_loss1 = bce_loss[:, 0, :, :].mean()
        bce_loss2 = bce_loss[:, 1, :, :].mean()
        bce_loss3 = bce_loss[:, 2, :, :].mean()
        bce_loss4 = bce_loss[:, 3, :, :].mean()

        dice_loss1, dice_loss2, dice_loss3, dice_loss4 = self.dice(input, target)
        bce_loss = self.alpha * (bce_loss1 + bce_loss2 + bce_loss4) + 0.5 * bce_loss3
        dice_loss = (1 - self.alpha) * (dice_loss1 + dice_loss2 + dice_loss4) + 0.5 * dice_loss3

        loss = (bce_loss + dice_loss) / 4

        return loss, bce_loss, dice_loss