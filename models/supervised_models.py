from models.cv_handler import calculate_metrics_cv
from visualization.plot_data import visualize_tree

import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


# TODO make return_train_score a parameter

def random_forest(x_train, y_train, cv, name="RandomForest", resample=False,
                  n_estimators=10, criterion="entropy", max_samples=0.6, max_features="sqrt", visualize=False, only_model=False, **kwargs):
    """ Random Forest.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices (not needed if testing is done)
        name (str): Name/Description for the model
        resample (bool): if True, a balanced Random Forest is used which randomly undersamples each bootstrap sample to balance it.
        n_estimators (int): how many trees the RF should have
        criterion (str): either "entropy" or "gini" in order to calculate most important attributes
        max_samples (float): between 0 and 1, how many of the data points should be sampled into one decision tree
        max_features (str): either "sqrt" or "log2" to determine number of features for each decision tree
        visualize (bool): whether a single decision tree from the forest should be plotted
        only_model (bool): if True returns only the model
    Returns:
        model or dict: only the model itself or a dict containing results of the crossvalidated models
    """
    # class_weight (str): should be at least 'balanced'. 'balanced_subsample' should be better.
    if resample:
        rf = BalancedRandomForestClassifier(n_estimators = n_estimators,
                                    criterion = criterion,
                                    bootstrap = True,
                                    max_samples = max_samples,     # 60 % of the training data (None: all)
                                    max_features = max_features, # uses sqrt(num_features) features
                                    class_weight = "balanced", # balanced_subsample computes weights based on bootstrap sample
                                    random_state = 42) #random state might not work
    else:
        rf = RandomForestClassifier(n_estimators = n_estimators,
                                    criterion = criterion,
                                    bootstrap = True,
                                    max_samples = max_samples,     # 60 % of the training data (None: all)
                                    max_features = max_features, # uses sqrt(num_features) features
                                    class_weight = "balanced", # balanced_subsample computes weights based on bootstrap sample
                                    random_state = 42)

    if visualize:
        visualize_tree(rf, x_train, y_train, file_name="plots/tree")

    if only_model:
        return rf
    # return rf, calculate_metrics_cv(model=rf, X=x_train, y_true=y_train, cv=cv, name=name)

    return calculate_metrics_cv(model=rf, X=x_train, y_true=y_train, cv=cv, name=name)

def svm(x_train, y_train, cv, C=0.95, decision_function_shape="ovr", kernel="rbf", gamma="auto", name="SupportVectorMachine", only_model=False, **kwargs):
    """ Support Vector Machine with Radial Basis functions as kernel.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        C (float): regularization parameter, controls directly overfitting. degree of allowed missclassifications. 1 means not missclassifications.
        decision_function_shape (str): "ovr" or "ovo" strategy for multiclass SVM
        kernel (str):  "linear", "poly", "rbf" or "sigmoid". For "poly" you should also set the degree.
        gamma (num or Str): gamma value or kernel coefficient for "rbf", "poly" and "sigmoid" kernel
        name (str): Name/Description for the model
        only_model (bool): if True returns only the model
    Returns:
        dict: contains results of models
    """
    svm = SVC(decision_function_shape = decision_function_shape,
              C = C,
              kernel = kernel,
              gamma = gamma,
              class_weight = "balanced",
              random_state = 24)

    if only_model:
        return svm

    return calculate_metrics_cv(model=svm, X=x_train, y_true=y_train, cv=cv, name=name)


# imbalanced data does not hurt knns
def knn(x_train, y_train, cv, n_neighbors=20, weights="distance", name="KNearestNeighbours", only_model=False, **kwargs):
    """ Support Vector Machine with Radial Basis functions as kernel.
    Parameters:
        x_train (pd.DataFrame): Input data for training
        y_train (pd.Series): Target data for training
        cv (list of tuples): cross validation indices
        n_neighbors (int): Number of neighbors to consider
        weights (str): either "distance" or "uniform". uniform is a simple majority vote.
            distance means that the nieghbours are weighted according to their distances.
        name (str): Name/Description for the model
        only_model (bool): if True returns only the model
    Returns:
        dict: contains results of models
    """
    knn = KNeighborsClassifier(n_neighbors = n_neighbors,
                               weights = "distance")
    if only_model:
        return knn

    return calculate_metrics_cv(model=knn, X=x_train, y_true=y_train, cv=cv, name=name)

# specifically for imbalanced data
# https://imbalanced-learn.org/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html
def ada_boost(x_train, y_train, cv, n_estimators=100, sampling_strategy="not majority", name="AdaBoost", only_model=False, **kwargs):
    """Bags AdaBoost learners which are trained on balanced bootstrap samples.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        n_estimators (int): number of boosted trees to consider
        sampling_strategy (str): "all", "not majority", "minority" and more. See docu of classifer for more details.
        name (str): Name/Description for the model
        only_model (bool): if True returns only the model
    Returns:
        dict: contains results of models
    """
    eec = EasyEnsembleClassifier(n_estimators=n_estimators,
                                 sampling_strategy=sampling_strategy,
                                 random_state=42)
    if only_model:
        return eec

    return calculate_metrics_cv(model=eec, X=x_train, y_true=y_train, cv=cv, name=name)
