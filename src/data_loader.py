import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from configure import SPLIT_FOLDER, DATA_FOLDER
import albumentations as albu

train_aug_seg = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0,
                          interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.5)
])

train_aug_cls = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15,
                          interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.5)
])

train_aug_cls_pseudo = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5)
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


class SteelDatasetSeg(Dataset):
    def __init__(self, df, phase):
        self.df = df
        self.data_folder = DATA_FOLDER
        self.phase = phase
        self.filenames = self.df.ImageId.values

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)

        image_path = os.path.join(self.data_folder, image_id)
        image = cv2.imread(image_path)

        if self.phase == "train":
            augmented = train_aug_seg(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            mask = (mask > 0.5).astype(np.float32)

        image = torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32)) / 255.0
        mask = torch.from_numpy(mask).permute(2, 0, 1)

        return image, mask

    def __len__(self):
        return len(self.filenames)


class SteelDatasetCls(Dataset):
    def __init__(self, df, phase):
        self.df = df
        self.data_folder = DATA_FOLDER
        self.phase = phase
        self.filenames = self.df.ImageId.values

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.data_folder, image_id)
        image = cv2.imread(image_path)

        if self.phase == "train":
            augmented = train_aug_cls(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            mask = (mask > 0.5).astype(np.float32)

        image = torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32)) / 255.0
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        label = (torch.sum(mask, (1, 2)) > 0).type(torch.float32)

        return image, label

    def __len__(self):
        return len(self.filenames)


class SteelDatasetClsPseudoLabels(Dataset):
    def __init__(self, df, df_pesudo, phase):
        self.df = df
        self.df_pesudo = df_pesudo
        self.data_folder = DATA_FOLDER
        self.phase = phase

    def __getitem__(self, idx):
        if idx < len(self.df):
            image_id, mask = make_mask(idx, self.df)
            image_path = os.path.join(self.data_folder, image_id)
            image = cv2.imread(image_path)

            if self.phase == "train":
                augmented = train_aug_cls(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
                mask = (mask > 0.5).astype(np.float32)

            image = torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32)) / 255.0
            mask = torch.from_numpy(mask).permute(2, 0, 1)
            label = (torch.sum(mask, (1, 2)) > 0).type(torch.float32)

        else:
            idx = idx - len(self.df)
            filename = self.df_pesudo.iloc[idx].ImageId
            image_path = os.path.join(self.data_folder, filename)
            image = cv2.imread(image_path)

            if self.phase == "train":
                image = train_aug_cls(image=image)['image']

            image = torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32)) / 255.0
            label = self.df_pesudo.iloc[idx][1:5].values.tolist()
            label = torch.from_numpy(np.array(label)).type(torch.float32)

        return image, label

    def __len__(self):
        return len(self.df) + len(self.df_pesudo)


def get_dataloader_seg(phase, fold, train_batch_size, valid_batch_size, num_workers):
    df_path = os.path.join(SPLIT_FOLDER, "fold_{}_{}.csv".format(fold, phase))
    df = pd.read_csv(df_path)

    df = df.loc[(df["defect1"] != 0) | (df["defect2"] != 0) | (df["defect3"] != 0) | (df["defect4"] != 0)]
    image_dataset = SteelDatasetSeg(df, phase)
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


def get_dataloader_cls(phase, fold, train_batch_size, valid_batch_size, num_workers):
    df_path = os.path.join(SPLIT_FOLDER, "fold_{}_{}.csv".format(fold, phase))
    df = pd.read_csv(df_path)

    image_dataset = SteelDatasetCls(df, phase)
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


def get_dataloader_cls_pesudo_labels(phase, fold, train_batch_size, valid_batch_size, num_workers):
    df_path = os.path.join(SPLIT_FOLDER, "fold_{}_{}.csv".format(fold, phase))
    df = pd.read_csv(df_path)

    if phase == "train":
        df_path_pesudo = os.path.join("../pseudo_labels/ResNet34.csv")
        df_pesudo = pd.read_csv(df_path_pesudo)
        image_dataset = SteelDatasetClsPseudoLabels(df, df_pesudo, phase)
        dataloader = DataLoader(image_dataset,
                                batch_size=valid_batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=True,
                                drop_last=True)

    else:
        image_dataset = SteelDatasetCls(df, phase)
        dataloader = DataLoader(image_dataset,
                                batch_size=valid_batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=False)

    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader_cls_pesudo_labels(phase="train",
                                                  fold=0,
                                                  train_batch_size=4,
                                                  valid_batch_size=4,
                                                  num_workers=1)

    for images, labels in dataloader:
        print(labels)
