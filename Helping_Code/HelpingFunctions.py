import numpy as np
import pandas as pd


def normalized_root_mean_square_error(y_true, y_pred):
    """
    Computes the Normalized Root Mean Square Error (NRMSE) between y_true and y_pred.
    Normalization is done using the range of y_true.

    Parameters:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_outputs)
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_outputs)

    Returns:
        float: NRMSE value
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    range_y = np.max(y_true) - np.min(y_true)
    return rmse / range_y if range_y != 0 else float('inf')


# Function to load DataSet
def load_dataset(file_path, num_attributes=2, num_classes=2):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 0 :  num_attributes].values
    y = data.iloc[:,  num_attributes:  num_attributes + num_classes].values
    # y = data.iloc[:, 9:10].values
    return X, y




# Function to get the indicies on different leaves
def get_leaf_samples(tree, X):
    """Return dictionary: leaf_id -> sample indices"""
    leaf_indices = tree.apply(X)
    leaf_samples = {}
    for i, leaf_id in enumerate(leaf_indices):
        if leaf_id not in leaf_samples:
            leaf_samples[leaf_id] = []
        leaf_samples[leaf_id].append(i)
    return {leaf: np.array(idxs) for leaf, idxs in leaf_samples.items()}