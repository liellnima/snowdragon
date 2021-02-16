from models.metrics import SCORERS, METRICS, METRICS_PROB, calculate_metrics_raw

import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from types import GeneratorType
from sklearn.model_selection import cross_validate


def assign_clusters(targets, clusters, cluster_num):
    """ Assigns snow labels to previously calculated clusters. This is a helper function for semisupervised models
    Parameters:
        targets: the int labels of our smp data - the target
        clusters: the cluster assignments from some unsupervised clustering algorithm
        cluster_num: the number of clusters
    Returns:
        list: with predicted snow labels for our data
    """
    pred_snow_labels = clusters
    for i in range(cluster_num):
        # filter for data with current cluster assignment
        mask = clusters == i
        # if there is something in this cluster
        if len(np.bincount(targets[mask])) > 0:
            # find out which snow grain label occurs most frequently
            snow_label = np.argmax(np.bincount(targets[mask]))
            # and assign it to all the data point belonging to the current cluster
            pred_snow_labels[mask] = snow_label
    # return the predicted snow labels
    return pred_snow_labels

def calculate_metrics_cv(model, X, y_true, cv, metrics=SCORERS, prob_metrics=METRICS_PROB, name=None, return_train_score=True):
    """ Calculate wished metrics one a cross-validation split for a certain data set and model
    Metrics calculated at the moment: Balanced accuracy, weighted recall, weighted precision
    Parameters:
        model: the model to fit from scikit learn
        X (nd array-like): the training data
        y_true (1d array-like): observed (true) target values for training data X
        cv (list): a k-fold list with (training, test) tuples containing the indices for training and test data
        metrics (dict): dictionary that contains scorer functions as values. Functions must contain the parameters y_true and y_pred.
        prob_metrics (dict): dictionary that contains probability based metrics (not as scorers!). Functions must contain the parameters y_true and y_pred.
        name (String): name of the model, if an entry for the scores list is wished (Default None)
        return_train_score (bool): if the training scores should be saved as well
    Returns:
        dict: with results
    """
    print("Cross-Validation Progress:")
    scores = cross_validate(model, X, y_true, cv=cv, scoring=metrics, return_train_score=return_train_score, n_jobs=-1, verbose=2)
    # in case model is not done suitable for probability prediction
    if hasattr(model, "probability"):
        model.probability = True
    prob_based_scores = prob_based_cross_validate(model, X, y_true, cv=cv, scoring=prob_metrics, return_train_score=return_train_score)
    all_scores = {**scores, **prob_based_scores}

    if name is not None:
        all_scores["model"] = name
    return all_scores

def prob_based_cross_validate(model, X, y_true, cv, scoring, return_train_score=True):
    """ Cross validation that can calculate probability based metrics.
    Parameters:
        model: the model to fit from scikit learn
        X (nd array-like): the training data
        y_true (1d array-like): observed (true) target values for training data X
        cv (list): a k-fold list with (training, test) tuples containing the indices for training and test data
        scoring (dict): contains the different (callable) functions which should be used as metrics. Must contain the parameters y_true and y_pred.
        return_train_score (bool): if the training scores should be saved as well
    Returns:
        dict: containing lists for different cv folds with the wished scores.
    """
    all_fit_time = []
    all_score_time = []
    # dictionary were scores are saved
    train_scores = {key: [] for key in scoring.keys()}
    valid_scores = {key: [] for key in scoring.keys()}

    # go through the different folds
    print("Probability Metrics Cross-Validation:")
    for k in tqdm(cv):
        # prepare training and validation data
        x_train = X.iloc[k[0]]
        y_train = y_true.iloc[k[0]]
        x_valid = X.iloc[k[1]]
        y_valid = y_true.iloc[k[1]]

        fit_time = time.time()
        model_fit = model.fit(x_train, y_train)
        all_fit_time.append(time.time() - fit_time)

        y_pred_train = model_fit.predict_proba(x_train)
        score_time = time.time()
        y_pred_valid = model_fit.predict_proba(x_valid)
        all_score_time.append(time.time() - score_time)

        # calculate the wished scores
        for key, metric in scoring.items():
            train_score = metric(y_true=y_train, y_pred=y_pred_train)
            valid_score = metric(y_true=y_valid, y_pred=y_pred_valid)
            train_scores[key].append(train_score)
            valid_scores[key].append(valid_score)

    # annotate train and validation dataset and transform lists to np.arrays
    train_scores = {("train_"+k): np.asarray(v) for k,v in train_scores.items()}
    valid_scores = {("test_"+k): np.asarray(v) for k,v in valid_scores.items()}

    # put scores together
    scores = {**train_scores, **valid_scores}
    scores["fit_time"] = np.asarray(all_fit_time)
    scores["score_time"] = np.asarray(all_score_time)

    return scores


# TODO make metrics/scorer parameter possible
# TODO make semi_supervised roc_auc and log_loss possible! (use predict_proba and the probability to express a class if in a certain cluster)
# TODO include flag to return train scores or not
def semisupervised_cv(model, unlabelled_data, x_train, y_train, cluster_num, cv, name=None):
    """ Crossvalidation for cluster-predict semisupervised approach.
    Parameters:
        model: model on which we fit our data
        unlabelled_data: complete unlabelled data (only features of interest)
        x_train: the input/features data where labels are available
        y_train: the target data for x_train containing the snow labels
        cluster_num (int): can be number of components, number of clusters or similar
        cv (list): list of tuples (train_indices, test_indices) for cv splitting (in case of reuse: must be a list, not a generator!)
        name (str): name of the model, default None
    Returns:
        dict: with scores for different metrics
    """
    # print warning if cv is a generator:
    if isinstance(cv, GeneratorType):
        warnings.warn("\nYour cross validation split cv is a generator. Consider handing over a list instead, if you want to reuse the generator.")
    all_train_y_pred = []
    all_train_y_true = []
    all_valid_y_pred = []
    all_valid_y_true = []
    all_fit_time = []
    all_score_time = []

    # cross validation
    print("Semisupervised Crossvalidation:")
    for k in tqdm(cv):
        # assignments
        train_target = y_train.iloc[k[0]]
        train_input  = x_train.iloc[k[0]]
        valid_target = y_train.iloc[k[1]]
        valid_input  = x_train.iloc[k[1]]

        # fitting
        fit_time = time.time()
        # concat unlabelled and labelled data and fit it
        fit_model = model.fit(pd.concat([unlabelled_data, train_input]))
        # shortcut:
        # fit_km = km.fit(train_input)
        all_fit_time.append(time.time() - fit_time)

        # predicting
        train_clusters = fit_model.predict(train_input)
        score_time = time.time()
        valid_clusters = fit_model.predict(valid_input)
        all_score_time.append(time.time() - score_time)

        # assign labels to clusters
        train_y_pred = assign_clusters(train_target, train_clusters, cluster_num)
        valid_y_pred = assign_clusters(valid_target, valid_clusters, cluster_num)
        all_train_y_pred.append(train_y_pred)
        all_valid_y_pred.append(valid_y_pred)

        # save true labels
        all_train_y_true.append(train_target)
        all_valid_y_true.append(valid_target)

    train_scores = calculate_metrics_raw(all_train_y_true, all_train_y_pred, name=name, annot="train")
    test_scores = calculate_metrics_raw(all_valid_y_true, all_valid_y_pred, name=name, annot="test")

    scores = {**train_scores, **test_scores}
    scores["fit_time"] = np.asarray(all_fit_time)
    scores["score_time"] = np.asarray(all_score_time)
    return scores


def mean_kfolds(scores):
    """ Produces the mean of scores resulting from scikit learn cross_validation function.
    Parameters:
        scores (dict): Dictionary of key - list pairs, where each list contains the results from the crossvalidation
    Returns:
        dict: same keys as dict scores, but the values are averaged now
    """
    return {key: scores[key].mean() if key != "model" else scores[key] for key, value in scores.items()}

# TODO: make this kind of stratified:
    # make the data a set of time-series data
    # one-hot-encode this data with labels (0 or 1: does this label occur in the timeseries?)
    # use scikit learn StratifiedKFold on this data set!
    # transfer this split to the smp_idx (which smp timeseries is used in which fold?)
def cv_manual(data, k):
    """ Performs a custom k-fold crossvalidation. Roughly 1/k % of the data is used a testing data,
    the rest is training data. This happens k times - each data chunk has been used as testing data once.

    Paramters:
        data (pd.DataFrame): data on which crossvalidation should be performed
        k (int): number of folds for cross validation
    Returns:
        list: iteratable list of length k with tuples of np 1-d arrays (train_indices, test_indices)
    """
    # assign each profile a number between 1 and 10
    cv = []
    profiles = list(data["smp_idx"].unique()) # list of all smp profiles of data
    k_idx = np.resize(np.arange(1, k+1), 69) # k indices for the smp profiles
    np.random.seed(42)
    np.random.shuffle(k_idx)

    # for each fold k, the corresponding smp profiles are used as test data
    for k in range(1, k+1):
        # indices for the current fold
        curr_fold_idx = np.argwhere(k_idx == k)
        # get the corresponding smp_idx
        curr_smps = [profiles[i[0]] for i in curr_fold_idx]
        # put the indices corresponding to the current smps into the cv data
        mask = data["smp_idx"].isin(curr_smps)
        valid_indices = data[mask].index.values
        train_indices = data[~mask].index.values
        cv.append((train_indices, valid_indices))

    return cv
