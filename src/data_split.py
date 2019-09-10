import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from configure import TRAIN_DF_PATH, SPLIT_FOLDER

df = pd.read_csv(TRAIN_DF_PATH)
df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
df['ClassId'] = df['ClassId'].astype(int)
df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
df['defect1'] = 1 - df[1].isnull().astype(int)
df['defect2'] = 1 - df[2].isnull().astype(int)
df['defect3'] = 1 - df[3].isnull().astype(int)
df['defect4'] = 1 - df[4].isnull().astype(int)

mskf = MultilabelStratifiedKFold(n_splits=5, random_state=42)

X = df.index
y1 = df['defect1'].values.tolist()
y2 = df['defect2'].values.tolist()
y3 = df['defect3'].values.tolist()
y4 = df['defect4'].values.tolist()
y = list()
for i in range(len(y1)):
    y.append([y1[i], y2[i], y3[i], y4[i]])

for i, (train_index, valid_index) in enumerate(mskf.split(X=X, y=y)):
    df_train, df_valid = df.iloc[train_index], df.iloc[valid_index]
    df_train.to_csv(os.path.join(SPLIT_FOLDER, "fold_{}_train.csv".format(i)))
    df_valid.to_csv(os.path.join(SPLIT_FOLDER, "fold_{}_valid.csv".format(i)))
