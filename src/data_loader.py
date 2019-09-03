import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize

from configure import SPLIT_FOLDER, DATA_FOLDER
from utils import img_to_tensor, mask_to_tensor


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
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')

    return filename, masks


class SteelDataset(Dataset):
    def __init__(self, df, phase, aug_prob=0.5):
        self.df = df
        self.data_folder = DATA_FOLDER
        self.phase = phase
        self.filenames = self.df.ImageId.values
        self.aug_prob = aug_prob
        self.normalizer = Normalize()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.data_folder, image_id)
        image = cv2.imread(image_path)

        if self.phase == "train":
            image, mask = self.transform(image=image, mask=mask)

        image, mask = img_to_tensor(image), mask_to_tensor(mask)
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return image, mask

    def __len__(self):
        return len(self.filenames)

    def transform(self, image, mask):
        # Random horizontal flipping
        if np.random.rand() < self.aug_prob:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Random vertical flipping
        if np.random.rand() < self.aug_prob:
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Random shift and and scale
        if np.random.rand() < self.aug_prob:
            aug = ShiftScaleRotate(p=1.0, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
            augmented = aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask


def get_dataloader(phase, fold, batch_size, num_workers):
    df_path = os.path.join(SPLIT_FOLDER, "fold_{}_{}.csv".format(fold, phase))
    df = pd.read_csv(df_path)
    image_dataset = SteelDataset(df, phase)
    shuffle = True if phase == "train" else False
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=shuffle)

    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader(phase="train", fold=0, batch_size=1, num_workers=1)

    imgs, masks = next(iter(dataloader))

    print(imgs.shape)  # batch * 3 * 256 * 1600
    print(masks)  # batch * 4 * 256 * 1600
