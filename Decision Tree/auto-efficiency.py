import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

# Set seed for reproducibility
np.random.seed(42)

# Read the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the data by removing rows with missing values
data = data[data['horsepower'] != '?']

# Define features and target variable
y = data['mpg']
X = data.drop(columns=['mpg', 'car name'], axis=1)
X['horsepower'] = X['horsepower'].astype(float)

# Custom train-test split function
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    data = pd.concat([X, y], axis=1)

    shuffled_data = data.sample(frac=1, random_state=random_state)

    split_index = int((1 - test_size) * len(data))

    train_data = shuffled_data.iloc[:split_index]
    test_data = shuffled_data.iloc[split_index:]

    X_train = train_data.iloc[:, :-1].reset_index(drop=True)
    y_train = train_data.iloc[:, -1].reset_index(drop=True)

    X_test = test_data.iloc[:, :-1].reset_index(drop=True)
    y_test = test_data.iloc[:, -1].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

# Split data into train and test sets
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

## Using our decision tree
my_tree = DecisionTree(criterion='information_gain')
my_tree.fit(X_train, y_train)
my_tree.plot()

y_hat = my_tree.predict(X_test)
print("\nRMSE using our tree = ", rmse(y_hat, y_test))
print("MAE using our tree = ", mae(y_hat, y_test))

## Using decision tree module from scikit-learn
decision_tree = DecisionTreeRegressor(max_depth=5)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(f"\nRMSE using tree module from scikit-learn = ", rmse(y_pred, y_test))
print(f"MAE using tree module from scikit-learn = ", mae(y_pred, y_test), "\n")

# Create a DataFrame to compare predictions
comparison_df = pd.concat([y_test.reset_index(drop=True), y_hat.reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True)], axis=1)
comparison_df.columns = ['y_test', 'our_tree', 'sklearn_tree']
print(comparison_df)
