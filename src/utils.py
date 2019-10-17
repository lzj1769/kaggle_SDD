import os
import numpy as np
import random
import torch


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def predict_to_onehot(predict, num_class=4):
    value, index = torch.max(predict, 1, keepdim=True)
    value = value.repeat(1, num_class, 1, 1)
    index = index.repeat(1, num_class, 1, 1)
    arange = torch.arange(1, num_class + 1).view(1, num_class, 1, 1).to(predict.device)
    onehot = (index == arange).float()
    value = value * onehot
    return value


def compute_dice(probability, truth, threshold=0.5):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channels = [0.] * channel_num
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(probability[i, j, :, :], truth[i, j, :, :], threshold)
                mean_dice_channels[j] += channel_dice / batch_size

    return mean_dice_channels


def dice_single_channel(probability, truth, threshold):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    if p.sum() == 0 and t.sum() == 0:
        dice = 1
    else:
        dice = (2.0 * (p * t).sum()) / (p.sum() + t.sum()).item()

    return dice


if __name__ == '__main__':
    from data_loader import get_dataloader
    from torch.nn import BCEWithLogitsLoss
    from model import UResNet34

    dataloader = get_dataloader(phase="train", fold=0, batch_size=4, num_workers=2)
    model = UResNet34()
    model.cuda()
    model.train()
    imgs, masks = next(iter(dataloader))
    imgs, masks = next(iter(dataloader))
    preds = model(imgs.cuda())
    criterion = BCEWithLogitsLoss()
    loss = criterion(preds, masks.cuda())
    print(loss.item())
