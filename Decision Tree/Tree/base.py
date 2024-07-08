from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *


class RealDiscreteNode:
    def __init__(self):
        self.attribute = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None
        self.data = None

    def display_tree_text(self, c, level=0):
        indent = "  " * level
        if self.left is None:
            if c == "r":
                print(f"{indent}|NO - {self.label}")
            else:
                print(f"{indent}|YES - {self.label}")
            return

        elif c == "r":
            print(f"{indent}|NO - (Data[{self.attribute}] <= {self.threshold}) :")
        else:
            if level == 0:
                print(f"(Data[{self.attribute}] <= {self.threshold}) :")
            else:
                print(f"{indent}|YES - (Data[{self.attribute}] <= {self.threshold}) :")

        self.left.display_tree_text("l", level + 1)
        self.right.display_tree_text("r", level + 1)


class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}

    def display_tree(self, val, level=0):
        indent = "    " * level
        if self.attribute is not None:
            print(indent + "|" + str(val) + ":  " + str(self.attribute) + "=>")
        else:
            print(indent + "|" + str(val) + ":  " + " Prediction: " + str(self.label))

        for child in self.children.keys():
            self.children[child].display_tree(child, level + 1)


class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.depth = 0
        self.root = None
        self.b1 = None
        self.b2 = None

    def the_real_fit(self, node, X: pd.DataFrame, y: pd.Series) -> None:
        if self.depth == 0:
            self.b1 = check_if_real(y)
            self.b2 = check_if_real(X.iloc[:, 1])

        if self.b2:
            root = node
            n, p = X.shape
            features = np.arange(p)

            if y.nunique() == 1:
                root.label = y.unique()[0]
                return

            if self.depth >= self.max_depth:
                if self.b1:
                    root.label = y.mean()
                else:
                    root.label = y.mode().iloc[0]
                return

            self.depth += 1

            root.attribute, root.threshold = best_split(X, y, p, n, self.b1, self.criterion)

            root.left = RealDiscreteNode()
            root.right = RealDiscreteNode()
            (d1, root.left.data), (d2, root.right.data) = split(X, y, root.attribute, root.threshold)

            if self.depth == 1:
                self.root = root

            self.the_real_fit(root.left, d1, root.left.data)
            self.the_real_fit(root.right, d2, root.right.data)
            self.depth -= 1

        else:
            n, p = X.shape
            features = X.columns

            root = node

            if y.nunique() == 1:
                root.label = y.unique()[0]
                return

            if len(features) == 0 or self.depth == self.max_depth:
                if self.b1:
                    root.label = y.mean()
                else:
                    root.label = y.mode().iloc[0]
                return

            self.depth += 1

            best_attribute = None
            best_info_gain = -np.inf
            for attribute in features:
                if not self.b1:
                    if self.criterion == "gini_index":
                        info_gain = information_gain_gini(X, y, attribute)
                    else:
                        info_gain = information_gain_entropy(X, y, attribute, self.b1)
                else:
                    info_gain = information_gain_entropy(X, y, attribute, self.b1)

                if info_gain > best_info_gain:
                    best_attribute = attribute
                    best_info_gain = info_gain

            root.attribute = best_attribute

            features_values = X[root.attribute].unique()
            for val in features_values:
                child = Node()
                root.children[val] = child
                sub_X = X[X[root.attribute] == val]
                sub_X = sub_X.drop(columns=root.attribute)
                sub_y = y[X[root.attribute] == val]
                if self.depth == 1:
                    self.root = root
                self.the_real_fit(child, sub_X, sub_y)

            self.depth -= 1

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.depth == 0:
            self.b1 = check_if_real(y)
            self.b2 = check_if_real(X.iloc[:, 1])
        if self.b2:
            node = RealDiscreteNode()
        else:
            node = Node()
        self.the_real_fit(node, X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.b2:
            y_pred = []
            for _, i in X.iterrows():
                node = self.root
                while node.label is None:
                    if i[node.attribute] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                y_pred.append(node.label)
            return pd.Series(y_pred)
        else:
            predictions = []

            for _, sample in X.iterrows():
                current_node = self.root

                n_iterations = 0
                while current_node.children and n_iterations < 100:
                    attribute_value = sample[current_node.attribute]

                    if attribute_value in current_node.children:
                        current_node = current_node.children[attribute_value]
                    n_iterations += 1
                predictions.append(current_node.label)

            return pd.Series(predictions)

    def plot(self):
        if self.b2:
            self.root.display_tree_text("l")
        else:
            self.root.display_tree("")
