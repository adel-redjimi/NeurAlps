import numpy as np

def binary_accuracy(y_true, y_pred):
    return np.mean(
        y_true.flatten() == (y_pred.flatten() >= .5).astype(int)
    )

def multi_class_accuracy(y_true, y_pred):
    return np.mean(
        np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
    )