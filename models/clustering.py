# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_preprocessing import idx_to_int
from data_handling.data_parameters import LABELS
from models.visualization import visualize_original_data # TODO or something like this
from models.metrics import METRICS
import models.metrics as my_metrics

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from tabulate import tabulate
import time
import warnings
from types import GeneratorType

# Other metrics: https://stats.stackexchange.com/questions/390725/suitable-performance-metric-for-an-unbalanced-multi-class-classification-problem
# TODO just import the metrics you need or everything
from sklearn import metrics
from sklearn.metrics import make_scorer, balanced_accuracy_score, recall_score, precision_score, roc_auc_score, log_loss

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# for over and undersampling
from imblearn.ensemble import EasyEnsembleClassifier

# from sklearn.multioutput import MultiOutputClassifier

# TODO plot confusion matrix beautifully (multilabel_confusion_matrix)

# TODO plot ROC AUC curve beautifully (roc_curve(y_true, y_pred))



def kmeans_old():

    # k-means clustering for one sample
    km = KMeans(n_clusters=5, init="random", n_init=10, random_state=42)

    clusters = km.fit_predict(sample[["mean_force", "var_force"]])
    print(clusters)

    sns.scatterplot(sample["var_force"], sample["mean_force"], hue=sample["label"], style=clusters).set_title("Clustering of S31H0369")
    plt.show()

    # we have in total 10 labels
    km_more = KMeans(n_clusters=10, init="random", n_init=100, random_state=42)

    clusters = km_more.fit_predict(smp_more[["mean_force", "var_force"]])
    print(clusters)

    sns.scatterplot(smp_more["var_force"], smp_more["mean_force"], hue=clusters).set_title("Variance and Mean force of 1000 samples")
    plt.show()

    # k-means clustering for all which are labelled

    # we have in total 10 labels
    km_lab = KMeans(n_clusters=10, init="random", n_init=100, random_state=42)

    clusters = km_lab.fit_predict(smp_labelled[["mean_force", "var_force"]])
    print(clusters)

    sns.scatterplot(smp_labelled["var_force"], smp_labelled["mean_force"], hue=smp_labelled["label"], style=clusters).set_title("Clustering for labelled data")
    plt.show()

def my_train_test_split(smp, test_size=0.2, train_size=0.8):
    """ Splits data into training and testing data
    Parameters:
        smp (df.DataFrame): Preprocessed smp data
        test_size (float): between 0 and 1, size of testing data
        train_size (float): between 0 and 1, size of training data
    Returns:
        quadruple: x_train, x_test, y_train, y_test
    """
    # labelled data
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]

    # print how many labelled profiles we have
    num_profiles = labelled_smp["smp_idx"].nunique()
    num_points = labelled_smp["smp_idx"].count()
    idx_list = labelled_smp["smp_idx"].unique()

    # sample randomly from the list
    train_idx, test_idx = train_test_split(idx_list, test_size=test_size, train_size=train_size, random_state=42)
    train = labelled_smp[labelled_smp["smp_idx"].isin(train_idx)]
    test = labelled_smp[labelled_smp["smp_idx"].isin(test_idx)]
    x_train = train.drop(["label"], axis=1)
    x_test = test.drop(["label"], axis=1)
    y_train = train["label"]
    y_test = test["label"]
    print("Labels in training data:\n", y_train.value_counts())
    print("Labels in testing data:\n", y_test.value_counts())
    return x_train, x_test, y_train, y_test

def majority_class_baseline(x_train, y_train, cv):
    y_preds_train = []
    y_trues_train = []
    y_preds_valid = []
    y_trues_valid = []
    all_fit_time = []
    all_score_time = []

    for k in cv:
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


    train_scores = calculate_metrics_raw(y_trues_train, y_preds_train, name="MajorityClassBaseline", annot="train")
    test_scores = calculate_metrics_raw(y_trues_valid, y_preds_valid, name="MajorityClassBaseline", annot="test")

    scores = {**train_scores, **test_scores}
    scores["fit_time"] = np.asarray(all_fit_time)
    scores["score_time"] = np.asarray(all_score_time)

    return scores

def calculate_metrics_raw(y_trues, y_preds, name=None, annot="train"):
    """ Calculates metrics when already the list of the observed and predicted target values is given. E.g. from a manual cross validation.
    Paramters:
        y_preds (list): List of lists (crossvalidation!) with predicted target values
        y_trues (list): List of listst (crossvalidation!) with true or observed target values
        name (String): Name of the model evaluated
        annot (String): indicates if we speak about train or test or validation data
    Returns:
        dict: dictionary with different scores
    """
    funcs = [my_metrics.balanced_accuracy, my_metrics.recall, my_metrics.precision]
    funcs_names = ["balanced_accuracy", "recall", "precision"]
    if annot is not None: annot = annot + "_"
    scores = {}
    if name is not None:
        scores["model"] = name
    # iterate through a list of metric functions and add the lists of results to scores
    for func, name in zip(funcs, funcs_names):
        scores[annot + name] = np.asarray([func(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)])

    return scores

# TODO I have to weight the labels! Assigning the most frequent label is not helpful! But that won't help either...
def assign_clusters(targets, clusters, cluster_num):
    """ Assigns snow labels to previously calculated clusters.
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

# TODO make metrics parameter possible
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
    for k in cv:
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

# https://towardsdatascience.com/cluster-then-predict-for-classification-tasks-142fdfdc87d6
def kmeans(unlabelled_data, x_train, y_train, cv, num_clusters=5, find_num_clusters="both", plot=True):
    """ Semisupervised kmeans algorithm. Assigns most frequent snow label to cluster.
    Parameters:
        unlabelled_data: Data on which the clustering should take place
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        num_clusters (int): number of clusters for kmeans, or maximum number of clusters
        find_num_clusters (str): either "sil" for Silhouette Coefficient or "acc" for balanced accuracy or "both".
            In case of "both" the optimal number of cluster is choosen according to results of acc
            Default: None - in this case only the kmeans model with num_clusters cluster is run
        plot (bool): whether silhouette coefficient or balanced accuracy should be plot
    Returns:
        dict: results from cross validation
    """
    if find_num_clusters is not None:
        max_cluster = num_clusters
        all_sil_scores = []
        bal_acc_scores = []
        # iterate through all possible numbers of clusters and calculate their sil score
        for cluster_num in range(2, max_cluster+1):
            print("Cluster Num: ", cluster_num)
            km = KMeans(n_clusters=cluster_num, init="random", n_init=cluster_num, random_state=42).fit(x_train)
            # calculate sil scores
            if find_num_clusters == "sil" or find_num_clusters == "both":
                sil_scores = metrics.silhouette_score(x_train, km.labels_, metric="euclidean")
                all_sil_scores.append(sil_scores)
            # calculate balanced accuracy
            if find_num_clusters == "acc" or find_num_clusters == "both":
                clusters = km.predict(x_train)
                y_pred = assign_clusters(y_train, clusters, cluster_num)
                bal_acc_scores.append(balanced_accuracy_score(y_train, y_pred))

        # find the argmax of the scores -> this is the perfect number of clusters
        # argmax of sil scores
        if find_num_clusters == "sil" or find_num_clusters == "both":
            sil_cluster_num_optimal = max(range(len(all_sil_scores)), key=lambda i: all_sil_scores[i]) + 2
            # the number of cluster which should be used: the optimal number of cluster
            # in case of "both" this will be overwritten by the balanced_acc maximum
            num_clusters = sil_cluster_num_optimal
        # argmax of bal_acc scores
        if find_num_clusters == "acc" or find_num_clusters == "both":
            acc_cluster_num_optimal = max(range(len(bal_acc_scores)), key=lambda i: bal_acc_scores[i]) + 2
            # the number of cluster which should be used: the optimal number of cluster
            num_clusters = acc_cluster_num_optimal

        # plot sil coefficients and balanced accuracy scores
        if plot:
            # plot for silhouette coefficient
            if find_num_clusters == "sil" or find_num_clusters == "both":
                plt.plot(range(2, max_cluster+1), all_sil_scores, label="Silhouette Coef")
                plt.axvline(sil_cluster_num_optimal, color="red", linestyle="--")
                plt.title("Silhouette Coefficient for K-means Clustering Model")
                plt.xlabel("Number of Clusters")
                plt.ylabel("Mean Silhouette Coefficient")
                plt.show()
            # plot for accuracy
            if find_num_clusters == "acc" or find_num_clusters == "both":
                plt.plot(range(2, max_cluster+1), bal_acc_scores, label="Balanced Acc")
                plt.axvline(acc_cluster_num_optimal, color="red", linestyle="--")
                plt.title("Balanced Accuracy for K-means Clustering Model")
                plt.xlabel("Number of Clusters")
                plt.ylabel("Balanced Accuracy")
                plt.show()

    km = KMeans(n_clusters=num_clusters, init="random", n_init=num_clusters, random_state=42)

    return semisupervised_cv(km, unlabelled_data, x_train, y_train, num_clusters, cv, name="Kmeans")

# TODO compare if bayesian gaussian mixture models are better than using BIC manually (the lower the better)
# TODO save plots
def gaussian_mix(unlabelled_data, x_train, y_train, cv, cov_type="tied", num_components=15, find_num_components="both", plot=True):
    """ Semisupervised Gaussian Mixture Algorithm. Assigns most frequent snow label to gaussians.
    Parameters:
        unlabelled_data: Data on which the clustering should take place
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        cov_type (str): type of covariance used for gaussian mixture model - one of: "tied", "diag", "spherical", "full"
        num_components (int): number of distributions (maximally) used for the model
        find_num_components (str): either "bic" for Bayesian Information Criterion or "acc" for balanced accuracy or "both".
            In case of "both" the optimal number of cluster is choosen according to results of acc
            Default: None - in this case only the kmeans model with num_clusters cluster is run
        plot (bool): whether the bic and balanced accuracy should be plot
    Returns:
        dict: results of crossvalidation
    """
    if find_num_components:
        max_components = num_components
        all_bic_scores = []
        bal_acc_scores = []

        for n_gaussians in range(1, max_components+1):
            print("N Components: ", n_gaussians)
            gm = GaussianMixture(n_components=n_gaussians, init_params="random", max_iter=150, covariance_type=cov_type, random_state=42)
            gm.fit(x_train)
            # calculate bic score
            if find_num_components == "bic" or find_num_components == "both":
                all_bic_scores.append(gm.bic(x_train))
            # calculate balanced accuracy
            if find_num_components == "acc" or find_num_components == "both":
                clusters = gm.predict(x_train)
                y_pred = assign_clusters(y_train, clusters, n_gaussians)
                bal_acc_scores.append(balanced_accuracy_score(y_train, y_pred))

        # optimal number of distributions is the one with the lowest bayesian information criterion or highest accuracy
        if find_num_components == "bic" or find_num_components == "both":
            bic_components_num_optimal = min(range(len(all_bic_scores)), key=lambda i: all_bic_scores[i]) + 1
            # in case of "both" this will be overwritten by the balanced_acc maximum
            n_components = bic_components_num_optimal
        # argmax of bal_acc scores
        if find_num_components == "acc" or find_num_components == "both":
            acc_components_num_optimal = max(range(len(bal_acc_scores)), key=lambda i: bal_acc_scores[i]) + 1
            n_components = acc_components_num_optimal

        if plot:
            if find_num_components == "bic" or find_num_components == "both":
                plt.plot(range(1, max_components+1), all_bic_scores)
                plt.axvline(bic_components_num_optimal, color="red", linestyle="--")
                plt.title("Bayesian Information Criterion for Gaussian Mixture Model, {}".format(cov_type))
                plt.xlabel("Number of Gaussian Distributions")
                plt.ylabel("BIC")
                plt.show()
            if find_num_components == "acc" or find_num_components == "both":
                plt.plot(range(1, max_components+1), bal_acc_scores)
                plt.axvline(acc_components_num_optimal, color="red", linestyle="--")
                plt.title("Balanced Accuracy for Gaussian Mixture Model, {}".format(cov_type))
                plt.xlabel("Number of Gaussian Distributions")
                plt.ylabel("Balanced Accuracy")
                plt.show()

    gm = GaussianMixture(n_components=n_components, init_params="random", max_iter=150, covariance_type=cov_type, random_state=42)
    return semisupervised_cv(gm, unlabelled_data, x_train, y_train, n_gaussians, cv, name="GaussianMixture_{}".format(cov_type))

def bayesian_gaussian_mix(unlabelled_data, x_train, y_train, cv, cov_type="tied", num_components=15):
    """ Semisupervised Variational Bayesian estimation of a Gaussian Mixture Algorithm. Assigns most frequent snow label to gaussians.
    Find automatically the right number of
    Parameters:
        unlabelled_data: Data on which the clustering should take place
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        cov_type (str): type of covariance used for gaussian mixture model - one of: "tied", "diag", "spherical", "full"
        num_components (int): number of distributions maximally used for the model
        plot (bool): whether the bic and balanced accuracy should be plot
    Returns:
        dict: results of crossvalidation
    """
    bgm = GaussianMixture(n_components=num_components, init_params="random", max_iter=150, covariance_type=cov_type, random_state=42)

    return semisupervised_cv(bgm, unlabelled_data, x_train, y_train, num_components, cv, name="BayesianGaussianMixture_{}".format(cov_type))

def self_training(unlabelled_data, x_train, y_train, cv):
    """ Self training
    """
    return "blubb"

def label_spreading(unlabelled_data, x_train, y_train, cv):
    """ Label spreading
    """
    return "blubb"


def calculate_metrics_cv(model, X, y_true, cv, name=None, return_train_score=True):
    """ Calculate wished metrics one a cross-validation split for a certain data set and model
    Metrics calculated at the moment: Balanced accuracy, weighted recall, weighted precision
    Parameters:
        model: the model to fit from scikit learn
        X (nd array-like): the training data
        y_true (1d array-like): observed (true) target values for training data X
        cv (list): a k-fold list with (training, test) tuples containing the indices for training and test data
        name (String): name of the model, if an entry for the scores list is wished (Default None)
        return_train_score (bool): if the training scores should be saved as well
    Returns:
        dict: with results
    """
    # TODO get roc_auc and log_loss running
    # TODO use METRICS instead (and make it a parameter!)
    metrics = {"balanced_accuracy": make_scorer(balanced_accuracy_score),
               "recall": make_scorer(recall_score, average="weighted"),
               "precision": make_scorer(precision_score, average="weighted")}
               #"roc_auc": make_scorer(roc_auc_score, average="weighted", multi_class="ovr")} # TODO check if ovo makes more sense
               #"log_loss": make_scorer(log_loss, greater_is_better=False)}
    scores = cross_validate(model, X, y_true, cv=cv, scoring=METRICS, return_train_score=return_train_score)
    if name is not None:
        scores["model"] = name
    return scores

# TODO make name a parameter and return_train_score as well
def random_forest(x_train, y_train, cv):
    """ Random Forest.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
    Returns:
        float: balanced_accuracy_score of training (for the moment)
    """
    rf = RandomForestClassifier(n_estimators=10,
                                criterion = "entropy",
                                bootstrap = True,
                                max_samples = 0.6,     # 60 % of the training data (None: all)
                                max_features = "sqrt", # uses sqrt(num_features) features
                                class_weight = "balanced", # balanced_subsample computes weights based on bootstrap sample
                                random_state = 42)
    return calculate_metrics_cv(model=rf, X=x_train, y_true=y_train, cv=cv, name="RandomForest")

def svm(x_train, y_train, cv, gamma="auto"):
    """ Support Vector Machine with Radial Basis functions as kernel.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        gamma (num or Str): gamma value for svm
    Returns:
        float: balanced_accuracy_score of training (for the moment)
    """
    svm = SVC(decision_function_shape = "ovr",
              kernel = "rbf",
              gamma = gamma,
              class_weight = "balanced",
              random_state = 24)
    return calculate_metrics_cv(model=svm, X=x_train, y_true=y_train, cv=cv, name="SupportVectorMachine")

# specifically for imbalanced data
def AdaBoost(x_train, y_train, cv):
    """Bags AdaBoost learners which are trained on balanced bootstrap samples.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
    Returns:
        float: balanced_accuracy_score of training (for the moment)
    """
    eec = EasyEnsembleClassifier(n_estimators=100,
                                 sampling_strategy="all",
                                 random_state=42)
    return calculate_metrics_cv(model=eec, X=x_train, y_true=y_train, cv=cv, name="AdaBoost")

# imbalanced data does not hurt knns
def knn(x_train, y_train, cv, n_neighbors):
    """ Support Vector Machine with Radial Basis functions as kernel.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        n_neighbors: Number of neighbors to consider
    Returns:
        float: balanced_accuracy_score of training (for the moment)
    """
    knn = KNeighborsClassifier(n_neighbors = n_neighbors,
                               weights = "distance")
    return calculate_metrics_cv(model=knn, X=x_train, y_true=y_train, cv=cv, name="KNearestNeighbours")

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

def sum_up_labels(smp, labels, name, label_idx):
    """ Sums up the datapoints belonging to one of the classes in labels to one class.
    Parameters:
        smp (pd.DataFrame): a dataframe with the smp profiles
        labels (list): a list of Strings with the labels which should be united
        name (String): name of the new unified class
        label_idx (int): the number identifying which label it is
    Returns:
        pd.DataFrame: the updated smp
    """
    int_labels = [LABELS[label] for label in labels]
    smp.loc[smp["label"].isin(int_labels), "label"] = label_idx
    # add a new key value pair to labels (and antilabels and colors?)
    # TODO update: add this also to antilabels and colors
    LABELS[name] = label_idx
    return smp

def mean_kfolds(scores):
    """ Produces the mean of scores resulting from scikit learn cross_validation function.
    Parameters:
        scores (dict): Dictionary of key - list pairs, where each list contains the results from the crossvalidation
    Returns:
        dict: same keys as dict scores, but the values are averaged now
    """
    return {key: scores[key].mean() if key != "model" else scores[key] for key, value in scores.items()}

def main():
    # 1. Load dataframe with smp data
    smp = load_data("smp_lambda_delta_gradient.npz")

    # prepare dataset of unlabelled data
    # TODO: fix this: CURRENTLY crushes for smp_lambda_delta_gradient
    unlabelled_smp = smp[(smp["label"] == 0)]
    # set unlabelled_smp label to -1
    unlabelled_smp.loc["label"] = -1
    unlabelled_smp_x = unlabelled_smp.drop(["label", "smp_idx"], axis=1)
    unlabelled_smp_x = unlabelled_smp_x.dropna()
    unlabelled_smp_y = pd.Series(np.repeat(-1, len(unlabelled_smp_x)))

    # TODO: maybe visualize some things only after normalization and standardization?
    # 2. Visualize the original data
    #visualize_original_data(smp)
    # 3. Sum up certain classes if necessary (alternative: do something to balance the dataset)
    smp = sum_up_labels(smp, ["df", "drift_end", "ifwp", "if", "sh"], name="rare", label_idx=18)

    # 4. Split up the data into training and test data
    x_train, x_test, y_train, y_test = my_train_test_split(smp)
    # reset internal panda index
    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # 5. Normalize and standardize the data
    # TODO

    # 6. Make crossvalidation split
    k = 3
    # Note: if we want to use StratifiedKFold, we can just hand over an integer to the functions
    cv_stratified = StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x_train, y_train)
    cv_stratified = list(cv_stratified)
    # yields a list of tuples with training and test indices
    cv = cv_manual(x_train, k) # in progress

    x_train = x_train.drop(["smp_idx"], axis=1)
    x_test = x_test.drop(["smp_idx"], axis=1)
    print(np.unique(y_train, return_counts=True))

    # 7. Call the models
    all_scores = []

    # # A Baseline - majority class predicition
    # baseline_acc = majority_class_baseline(x_train, y_train, cv_stratified)
    # all_scores.append(mean_kfolds(baseline_acc))
    #
    # # B kmeans clustering (does not work)
    # kmeans_acc = kmeans(unlabelled_smp_x, x_train, y_train, cv_stratified, num_clusters=30, find_num_clusters="both", plot=False)
    # all_scores.append(mean_kfolds(kmeans_acc))
    # # print(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))
    # # exit(0)
    #
    # # C mixture model clustering ("diag" works best at the moment)
    # gm_acc_diag = gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag", plot=False)
    # all_scores.append(mean_kfolds(gm_acc_diag))
    # bgm_acc_diag = bayesian_gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag")
    # all_scores.append(mean_kfolds(bgm_acc_diag))

    # # ARE TAKING TOO MUCH TIME
    # # D + E -> different data preparation necessary
    # # include unlabelled data points in x_train and y_train
    # x_train_all = pd.concat([x_train, unlabelled_smp_x])
    # y_train_all = pd.concat([y_train, unlabelled_smp_y])
    #
    # # D label spreading model
    # ls_model = LabelSpreading(kernel="knn", alpha=0.2, n_jobs=-1).fit(x_train_all, y_train_all)
    # y_pred = ls_model.predict(x_train_all)
    # ls_bal_acc = balanced_accuracy_score(y_pred, y_train_all)
    # print("Label Spreading Model, Training Accuracy: ", ls_bal_acc)
    # exit(0)
    #
    # # TODO it makes sense to use the best hyperparameter tuned models here!
    # # E self training model
    # svm = SVC(probability=True, gamma="auto")
    # st_model = SelfTrainingClassifier(svm, verbose=True).fit(x_train_all, y_train_all)
    # print("Hello3")
    # y_pred = st_model.predict(x_train_all)
    # st_bal_acc = balanced_accuracy_score(y_pred, y_train_all)
    # print("Self Training Classifier, Training Accuracy: ", st_bal_acc)
    #
    #
    # print(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))


    # F random forests (works)
    rf_acc = random_forest(x_train, y_train, cv_stratified)
    all_scores.append(mean_kfolds(rf_acc))
    print(all_scores)
    exit(0)

    # G Support Vector Machines
    # works with very high gamma (overfitting) -> "auto" yields 0.75, still good and no overfitting
    svm_acc = svm(x_train, y_train, cv, gamma="auto")
    all_scores.append(mean_kfolds(svm_acc))

    # H knn (works with weights=distance)
    knn_acc = knn(x_train, y_train, cv, n_neighbors=20)
    all_scores.append(mean_kfolds(knn_acc))

    # I adaboost
    ada_acc = AdaBoost(x_train, y_train, cv)
    all_scores.append(mean_kfolds(ada_acc))

    # J LSTM

    # K Encoder-Decoder

    # 8. Visualize the results
    print(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))
    with open('plots/tables/models_with_baseline.txt', 'w') as f:
        f.write(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))



if __name__ == "__main__":
    main()
