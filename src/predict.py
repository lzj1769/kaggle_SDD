import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model import UResNet34

TEST_FOLDER = "../input/severstal-steel-defect-detection/test_images"
TEST_DF = "../input/severstal-steel-defect-detection/sample_submission.csv"
SAVE_MODEL_PATH = "../input/models/UResNet34"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def do_horizontal_flip(image):
    # flip left-right
    image = cv2.flip(image, 1)
    return image


def do_normalization(image, max_pixel_value=255):
    mean = np.array(MEAN, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(STD, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = image.astype(np.float32)
    img -= mean
    img *= denominator

    return img


def img_to_tensor(img):
    tensor = torch.from_numpy(np.expand_dims(np.moveaxis(img, -1, 0), 0).astype(np.float32))
    return tensor


def post_process(probability, threshold, min_size=3500):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class TestDataset(Dataset):
    def __init__(self, test_folder, test_df):
        self.test_folder = test_folder
        self.df = pd.read_csv(test_df)
        self.df['ImageId'] = self.df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.filename = self.df['ImageId'].unique().tolist()
        self.num_samples = len(self.filename)

    def __getitem__(self, idx):
        filename = self.filename[idx]
        path = os.path.join(self.test_folder, filename)
        image = cv2.imread(path)
        return filename, image

    def __len__(self):
        return self.num_samples


def main():
    model = UResNet34()
    model.cuda()
    model_save_path = os.path.join(SAVE_MODEL_PATH, "UResNet34_fold_0.pt")

    model.eval()
    state = torch.load(model_save_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    predictions = []
    test_dataset = TestDataset(test_folder=TEST_FOLDER, test_df=TEST_DF)
    for filename, image in test_dataset:
        image = do_normalization(image)
        image = img_to_tensor(image)
        preds = torch.sigmoid(model(image.cuda()))
        preds = preds.detach().cpu().numpy()
        for cls, pred in enumerate(preds[0]):
            pred, num = post_process(pred, 0.5)
            rle = mask2rle(pred)
            name = filename + f"_{cls + 1}"
            predictions.append([name, rle])



if __name__ == '__main__':
    main()
