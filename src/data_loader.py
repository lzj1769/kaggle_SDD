import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import normalize

from configure import SPLIT_FOLDER, DATA_FOLDER
import albumentations as albu

train_aug_seg = albu.Compose([
    albu.RandomCrop(height=128, width=800, p=1.0),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15,
                          interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.5)
])

train_aug_cls = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15,
                          interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.5)
])


def make_mask(row_id, df):
    # Given a row index, return image_id and mask (256, 1600, 4)
    filename = df.iloc[row_id].ImageId
    labels = df.iloc[row_id][1:5]

    masks = np.zeros((256, 1600, 4), dtype=np.float32)  # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)
    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                pos -= 1  # https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')

    return filename, masks


class SteelDataset(Dataset):
    def __init__(self, df, phase, task):
        self.df = df
        self.data_folder = DATA_FOLDER
        self.phase = phase
        self.task = task
        self.filenames = self.df.ImageId.values

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.data_folder, image_id)
        image = cv2.imread(image_path)

        if self.phase == "train":
            if self.task == "seg":
                augmented = train_aug_seg(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            elif self.task == "cls":
                augmented = train_aug_cls(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            else:
                raise "unknown task: {}".format(self.task)

        image = torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32)) / 255.0
        image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(mask).permute(2, 0, 1)

        return image, mask

    def __len__(self):
        return len(self.filenames)


def get_dataloader(phase, fold, train_batch_size, valid_batch_size, num_workers, task):
    df_path = os.path.join(SPLIT_FOLDER, "fold_{}_{}.csv".format(fold, phase))
    df = pd.read_csv(df_path)
    # select the image with non-empty masks for segmentation training
    if task == "seg":
        df = df.loc[(df["defect1"] != 0) | (df["defect2"] != 0) | (df["defect3"] != 0) | (df["defect4"] != 0)]

    image_dataset = SteelDataset(df, phase, task=task)
    shuffle = True if phase == "train" else False
    drop_last = True if phase == "train" else False
    batch_size = train_batch_size if phase == "train" else valid_batch_size
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=shuffle,
                            drop_last=drop_last)

    return dataloader
