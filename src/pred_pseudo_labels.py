# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from model import *

TEST_FOLDER = "../input/severstal-steel-defect-detection/test_images"
TEST_DF = "../input/severstal-steel-defect-detection/sample_submission.csv"


def do_horizontal_flip(image):
    # flip left-right
    image = cv2.flip(image, 1)
    return image


def img_to_tensor(img):
    tensor = torch.from_numpy(np.expand_dims(np.moveaxis(img, -1, 0), 0).astype(np.float32))
    return tensor


class TestDataset(Dataset):
    def __init__(self, test_folder, test_df):
        self.test_folder = test_folder
        self.df = pd.read_csv(test_df)
        self.df['ImageId'] = self.df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.filenames = self.df['ImageId'].unique().tolist()
        self.num_samples = len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.test_folder, filename)
        image = cv2.imread(path) / 255.0

        return filename, image

    def __len__(self):
        return self.num_samples


def predict_classification(image, model):
    image_raw = img_to_tensor(image)
    preds_raw = torch.sigmoid(model(image_raw.cuda()))
    preds_raw = preds_raw.detach().cpu().numpy()

    image_hflip = img_to_tensor(do_horizontal_flip(image))
    preds_hflip = torch.sigmoid(model(image_hflip.cuda()))
    preds_hflip = preds_hflip.detach().cpu().numpy()

    preds = (preds_raw + preds_hflip) / 2
    return preds


def pseudo_label_cls_s1(dataset, low_bound, up_bound):
    # predict defect vs. no defect
    pred_cls = np.zeros(shape=(len(dataset), 4), dtype=np.float32)
    for i in range(5):
        model = ResNet34(pretrained=False)
        model.cuda()
        model_save_path = "../input/models/ResNet34/ResNet34_fold_{}.pt".format(i)
        state = torch.load(model_save_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        model.eval()

        for j, (filename, image) in enumerate(dataset):
            pred_cls[j] += predict_classification(image=image, model=model)[0]

    for i in range(5):
        model = ResNet18(pretrained=False)
        model.cuda()
        model_save_path = "../input/models/ResNet18/ResNet18_fold_{}.pt".format(i)
        state = torch.load(model_save_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        model.eval()

        for j, (filename, image) in enumerate(dataset):
            pred_cls[j] += predict_classification(image=image, model=model)[0]

    pred_cls /= 5
    prediction = []
    for i, (filename, image) in enumerate(dataset):
        is_pseudo_label = True
        for cls in range(4):
            if low_bound < pred_cls[i, cls] < up_bound:
                is_pseudo_label = False

        if is_pseudo_label:
            defect1 = 0 if pred_cls[i, 0] < low_bound else 1
            defect2 = 0 if pred_cls[i, 1] < low_bound else 1
            defect3 = 0 if pred_cls[i, 2] < low_bound else 1
            defect4 = 0 if pred_cls[i, 3] < low_bound else 1
            prediction.append([filename, defect1, defect2, defect3, defect4,
                               pred_cls[i, 0], pred_cls[i, 1], pred_cls[i, 2], pred_cls[i, 3]])

    df = pd.DataFrame(prediction, columns=['ImageId',
                                           'defect1', 'defect2', 'defect3', 'defect4',
                                           'prob1', 'prob2', 'prob3', 'prob4'])

    df.to_csv("../pseudo_labels/PseudoLabels_S1_{}_{}.csv".format(low_bound, up_bound), index=False)


def pseudo_label_cls_s2(dataset, low_bound, up_bound):
    # predict defect vs. no defect
    pred_cls = np.zeros(shape=(len(dataset), 4), dtype=np.float32)
    for i in range(5):
        model = ResNet34(pretrained=False)
        model.cuda()
        model_save_path = "../input/models/ResNet34WithPseudoLabelsS1/ResNet34WithPseudoLabelsS1_fold_{}.pt".format(i)
        state = torch.load(model_save_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        model.eval()

        for j, (filename, image) in enumerate(dataset):
            pred_cls[j] += predict_classification(image=image, model=model)[0]

    for i in range(5):
        model = ResNet34(pretrained=False)
        model.cuda()
        model_save_path = "../input/models/ResNet18WithPseudoLabelsS1/ResNet18WithPseudoLabelsS1_fold_{}.pt".format(i)
        state = torch.load(model_save_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        model.eval()

        for j, (filename, image) in enumerate(dataset):
            pred_cls[j] += predict_classification(image=image, model=model)[0]

    pred_cls /= 10
    prediction = []
    for i, (filename, image) in enumerate(dataset):
        is_pseudo_label = True
        for cls in range(4):
            if low_bound < pred_cls[i, cls] < up_bound:
                is_pseudo_label = False

        if is_pseudo_label:
            defect1 = 0 if pred_cls[i, 0] < low_bound else 1
            defect2 = 0 if pred_cls[i, 1] < low_bound else 1
            defect3 = 0 if pred_cls[i, 2] < low_bound else 1
            defect4 = 0 if pred_cls[i, 3] < low_bound else 1
            prediction.append([filename, defect1, defect2, defect3, defect4,
                               pred_cls[i, 0], pred_cls[i, 1], pred_cls[i, 2], pred_cls[i, 3]])

    df = pd.DataFrame(prediction, columns=['ImageId',
                                           'defect1', 'defect2', 'defect3', 'defect4',
                                           'prob1', 'prob2', 'prob3', 'prob4'])

    df.to_csv("../pseudo_labels/PseudoLabels_S2_{}_{}.csv".format(low_bound, up_bound), index=False)


if __name__ == '__main__':
    test_dataset = TestDataset(test_folder=TEST_FOLDER, test_df=TEST_DF)
    pseudo_label_cls_s1(test_dataset, low_bound=0.1, up_bound=0.9)
