import os
import cv2
import numpy as np
import random
import torch
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def do_horizontal_flip(image):
    image = cv2.flip(image, 1)
    return image


def do_vertical_flip(image):
    image = cv2.flip(image, 0)
    return image


def img_to_tensor(img):
    tensor = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))
    return tensor


def mask_to_tensor(mask):
    return torch.from_numpy(mask)


def compute_dice(preds, truth, threshold=0.5):
    probability = torch.sigmoid(preds)
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(probability[i, j, :, :], truth[i, j, :, :], threshold)
                mean_dice_channel += channel_dice / (batch_size * channel_num)

    return mean_dice_channel


def dice_single_channel(probability, truth, threshold, eps=1E-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return dice


def visualize(image, mask, original_image=None, original_mask=None):
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=18)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=18)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=18)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=18)


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
