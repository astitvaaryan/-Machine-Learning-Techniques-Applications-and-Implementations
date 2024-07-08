from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    correct_predictions = (y_hat == y).sum()
    total_predictions = y.size
    return correct_predictions / total_predictions

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    total_good_pred = np.count_nonzero(y_hat == 1)  # Number of times y_hat=1
    count = 0
    for i in range(len(y_hat)):
        if y_hat[i] == 1 and y[i] == 1:
            count += 1
    prec = count / total_good_pred
    return prec

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    total_good_pred = np.count_nonzero(y == 1)  # Number of times y_hat=1
    count = 0
    for i in range(len(y_hat)):
        if y_hat[i] == 1 and y[i] == 1:
            count += 1
    prec = count / total_good_pred
    return prec

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error (rmse)
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    if y_hat.dtype.name == 'category':
        y_hat = y_hat.cat.codes
    if y.dtype.name == 'category':
        y = y.cat.codes

    rms = np.sqrt(np.mean((y_hat - y) ** 2))
    return rms

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error (mae)
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    if y_hat.dtype.name == 'category':
        y_hat = y_hat.cat.codes
    if y.dtype.name == 'category':
        y = y.cat.codes

    mean_err = np.mean(np.abs(y_hat - y))
    return mean_err
