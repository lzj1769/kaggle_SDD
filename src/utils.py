import numpy as np

import warnings

warnings.simplefilter("ignore")


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

#
# if __name__ == '__main__':
#     masks = torch.rand((10, 4, 256, 1600))
#     preds = torch.rand((10, 4, 256, 1600))
#     print(metric(preds, masks))
