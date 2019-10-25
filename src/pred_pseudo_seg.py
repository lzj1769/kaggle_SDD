import pandas as pd

df_pseudo_cls = pd.read_csv("../pseudo_labels/PseudoLabels_S2_0.1_0.9.csv")
df_best_submission = pd.read_csv("../pseudo_labels/submission_0.92143.csv")

df_pseudo_cls = df_pseudo_cls.loc[
    (df_pseudo_cls["defect1"] != 0) | (df_pseudo_cls["defect2"] != 0) | (df_pseudo_cls["defect3"] != 0) | (
            df_pseudo_cls["defect4"] != 0)]

df_best_submission['ImageId'], df_best_submission['ClassId'] = zip(
    *df_best_submission['ImageId_ClassId'].str.split('_'))
df_best_submission['ClassId'] = df_best_submission['ClassId'].astype(int)
df_best_submission = df_best_submission.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

df_best_submission['defect1'] = 1 - df_best_submission[1].isnull().astype(int)
df_best_submission['defect2'] = 1 - df_best_submission[2].isnull().astype(int)
df_best_submission['defect3'] = 1 - df_best_submission[3].isnull().astype(int)
df_best_submission['defect4'] = 1 - df_best_submission[4].isnull().astype(int)

df_best_submission = df_best_submission.loc[df_pseudo_cls['ImageId'].values.tolist()]

df_best_submission.to_csv("../pseudo_labels/PseudoLabels_seg_S2.csv")
