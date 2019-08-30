import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from configure import SPLIT_FOLDER, DATA_FOLDER
from transform import *


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

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.data_folder, image_id)
        img = cv2.imread(image_path) / 255.0

        if self.phase == "train":
            img, mask = self.transform(img=img, mask=mask)

        img, mask = img_to_tensor(img), mask_to_tensor(mask)
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.filenames)

    def transform(self, img, mask):
        # Random horizontal flipping
        if np.random.rand() < self.aug_prob:
            img, mask = do_horizontal_flip(img), do_horizontal_flip(mask)

        # Random vertical flipping
        if np.random.rand() < self.aug_prob:
            img, mask = do_vertical_flip(img), do_vertical_flip(mask)

        # Random adjust brightness
        if np.random.rand() < self.aug_prob:
            c = np.random.choice(2)
            if c == 0:
                img = do_brightness_shift(img, np.random.uniform(-0.1, +0.1))
            if c == 1:
                img = do_brightness_multiply(img, np.random.uniform(1 - 0.08, 1 + 0.08))
        #
        # # Random shift and crop
        # if np.random.rand() < 0.5:
        #     c = np.random.choice(2)
        #     if c == 0:
        #         image, mask = do_random_shift_scale_crop_pad2(img, mask, 0.2)
        #
        #     if c == 1:
        #         image, mask = do_shift_scale_rotate2(img, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15))

        return img, mask


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
    dataloader = get_dataloader(phase="train", fold=1, batch_size=10, num_workers=1)

    imgs, masks = next(iter(dataloader))

    print(imgs)  # batch * 3 * 256 * 1600
    print(masks.shape)  # batch * 4 * 256 * 1600
