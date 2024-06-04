from models.metrics import calculate_metrics_raw

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

def fit_baseline(y_train):
    """ Returns majority class of y_train.
    Parameters:
        y_train (pd.Series): labels of training data
    Returns:
        num or str: the label that occurs most often
    """
    return y_train.mode()[0]

def predict_baseline(majority_fit, x_test):
    """ Returns a prediction for the testing data. (Majority vote during training).
    Parameters:
        majority_fit (num or str): from fit_baseline - the majority vote
        x_test (pd.Dataframe): the testing data
    Returns:
        np.ndarray: contains the predictions
    """
    return np.repeat(majority_fit, len(x_test))

def majority_class_baseline(x_train, y_train, cv, name="MajorityClassBaseline", **kwargs):
    """ A model which always predicts the majority class of the data it was fit on to.
    Parameters:
        x_train: Data samples and features on which the model should train.
        y_train: The corresponding labels to x_train
        cv (list): The training-validation split (indices of data in a list of tuples)
        name (str): Name/Description for the model
    Returns:
        dict: scores describing the performance of the model. Probability based measures cannot be calculated
    """
    y_preds_train = []
    y_trues_train = []
    y_preds_valid = []
    y_trues_valid = []
    all_fit_time = []
    all_score_time = []

    print("Crossvalidation of Baseline Model:")
    for k in tqdm(cv):
        # current target values for this fold (training and validation)
        fit_time = time.time()
        fold_y_train = y_train[k[1]]
        all_fit_time.append(time.time() - fit_time)
        fold_y_valid = y_train[k[0]]
        # append true labels for current fold (both for training and validation data)
        y_trues_train.append(fold_y_train)
        y_trues_valid.append(fold_y_valid)
        # majority class in this fold of training data (will be also the majority class for the validation set!)
        score_time = time.time()
        maj_class = fold_y_train.mode()
        y_pred = pd.Series(np.repeat(maj_class, len(fold_y_valid))) # predicted labels of validation data
        all_score_time.append(time.time() - score_time)

        y_preds_valid.append(y_pred)
        y_preds_train.append(pd.Series(np.repeat(maj_class, len(fold_y_train)))) # predicted labels of training data


    train_scores = calculate_metrics_raw(y_trues_train, y_preds_train, name=name, annot="train")
    test_scores = calculate_metrics_raw(y_trues_valid, y_preds_valid, name=name, annot="test")

    scores = {**train_scores, **test_scores}
    scores["fit_time"] = np.asarray(all_fit_time)
    scores["score_time"] = np.asarray(all_score_time)

    return scores
