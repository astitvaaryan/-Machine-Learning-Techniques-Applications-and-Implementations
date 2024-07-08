import numpy as np
import pandas as pd
from pandas.core.common import flatten


# Function to calculate the Mean Squared Error
def mse(y):
    m = y.mean()
    ms = ((y - m) ** 2).mean()
    return ms


# Function to calculate the information gain for real-valued input
def info_gain_real(y):
    n = len(y)
    total_ele = []

    for i in y:
        ele = i.values
        total_ele.append(ele)

    total_ele = list(flatten(total_ele))

    initial_mean = np.mean(total_ele)
    init_mse = np.mean((total_ele - initial_mean) ** 2)

    num_total_ele = np.shape(total_ele)[0]
    entropy_real = 0

    for i in y:
        ele = i.values
        ele = np.array(ele)
        mean = np.mean(ele)
        mse = np.mean((ele - mean) ** 2)
        weight = ele.shape[0] / num_total_ele
        entropy_real += weight * mse

    info_real = init_mse - entropy_real
    return info_real


# Function to split the dataset based on a given attribute and threshold
def split(X: pd.DataFrame, y: pd.Series, attribute, value):
    dataset1 = X.where(X[attribute] <= value).dropna()
    dataset2 = X.where(X[attribute] > value).dropna()

    y1 = y[dataset1.index]
    y2 = y[dataset2.index]

    return (dataset1, y1), (dataset2, y2)


# Function to find the best attribute and threshold for splitting
def best_split(X: pd.DataFrame, y: pd.Series, P: int, N: int, b1, criterion):
    best_split_info = {}
    max_info_gain = -np.inf

    for i in range(P):
        attribute = X[X.columns[i]]
        threshhold = np.unique(attribute)
        mean_threshhold = []

        for k in range(N - 1):
            mean_threshhold.append(attribute.iloc[k:k + 2].mean())

        for j in mean_threshhold:
            (dataset1, y1), (dataset2, y2) = split(X, y, X.columns[i], j)

            if b1:
                info_gain = info_gain_real([y1, y2])
            else:
                if criterion == "information_gain":
                    w1 = len(y1) / len(y)
                    w2 = len(y2) / len(y)
                    info_gain = entropy(y) - (entropy(y1) * w1 + entropy(y2) * w2)
                else:
                    w1 = len(y1) / len(y)
                    w2 = len(y2) / len(y)
                    info_gain = gini(y) - (gini(y1) * w1 + gini(y2) * w2)

            if info_gain >= max_info_gain:
                max_info_gain = info_gain
                best_split_info['attribute'] = X.columns[i]
                best_split_info['threshhold'] = j
                best_split_info['dataset1'] = dataset1
                best_split_info['dataset2'] = dataset2

    return best_split_info['attribute'], best_split_info['threshhold']


# Function to check if the output is real-valued
def check_if_real(y: pd.Series, tolerance=1e-10) -> bool:
    unique_values = y.unique()

    for val in unique_values:
        if len(unique_values) / len(y) >= 0.2:
            if not np.isclose(val, round(val), atol=tolerance):
                return True  # indicating real output

    return False  # indicating discrete output


# Function to calculate the Gini impurity
def gini(series):
    value_counts = series.value_counts(normalize=True)
    gini = 1 - np.sum((value_counts * np.log2(value_counts + 1e-6)) ** 2)
    return gini


# Function to calculate the entropy
def entropy(series):
    value_counts = series.value_counts(normalize=True)
    entropy = -np.sum(value_counts * np.log2(value_counts + (10 ** (-5))))
    return entropy


# Function to calculate the information gain for discrete input
def information_gain_entropy(df, Y: pd.Series, attribute, b1) -> float:
    if b1:
        total_entropy = mse(Y)
    else:
        total_entropy = entropy(Y)

    weighted_entropy = 0.0
    col = df[attribute]
    unique_values = col.unique()

    for value in unique_values:
        subset_Y = Y[df[attribute] == value]
        weight = len(subset_Y) / len(Y)

        if b1:
            weighted_entropy += weight * mse(subset_Y)
        else:
            weighted_entropy += weight * entropy(subset_Y)

    information_gain = total_entropy - weighted_entropy
    return information_gain


# Function to beautify the code
def information_gain_gini(df, Y: pd.Series, attribute) -> float:
    total_gini = gini(Y)
    weighted_gini = 0

    col = df[attribute]
    unique_values = col.unique()

    for value in unique_values:
        subset_Y = Y[df[attribute] == value]
        weight = len(subset_Y) / len(Y)
        weighted_gini += weight * gini(subset_Y)

    information_gain = total_gini - weighted_gini
    return information_gain
