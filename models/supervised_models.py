from models.cv_handler import calculate_metrics_cv
from models.visualization import visualize_tree
from models.visualization import smp_labelled
from models.helper_funcs import reverse_normalize

import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


def testing(model, x_train, y_train, x_test, y_test, smp_idx_train, smp_idx_test):
    """ Performs testing on a model. Model is fit on training data and evaluated on testing data. Prediction inklusive.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    for smp_name in smp_idx_test.unique():
        smp = pd.DataFrame({"mean_force": x_test["mean_force"], "distance": x_test["distance"], "label": y_test, "smp_idx": smp_idx_test})
        smp = reverse_normalize(smp, "mean_force", min=0, max=45)
        smp = reverse_normalize(smp, "distance", min=0, max=1187)
        smp.info()
        smp_wanted = smp[smp["smp_idx"] == smp_name]

        smp_labelled(smp_wanted, smp_name)
        smp_pred = smp.copy()
        smp_pred["label"] = y_pred
        smp_wanted_pred = smp_pred[smp_pred["smp_idx"] == smp_name]
        smp_labelled(smp_wanted_pred, smp_name)

    exit(0)


    # pick out a certain smp profile in the test set:


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
        model or (model, dict): tuple of the model itself and a dict containing results of models (or returns only model if indicated)
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
        visualize_tree(rf, x_train, y_train, file_name="plots/forests/tree")

    if only_model:
        return rf
    # return rf, calculate_metrics_cv(model=rf, X=x_train, y_true=y_train, cv=cv, name=name)

    return calculate_metrics_cv(model=rf, X=x_train, y_true=y_train, cv=cv, name=name)

def svm(x_train, y_train, cv, C=0.95, decision_function_shape="ovr", kernel="rbf", gamma="auto", name="SupportVectorMachine", **kwargs):
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
    Returns:
        dict: contains results of models
    """
    svm = SVC(decision_function_shape = decision_function_shape,
              C = C,
              kernel = kernel,
              gamma = gamma,
              class_weight = "balanced",
              random_state = 24)
    return calculate_metrics_cv(model=svm, X=x_train, y_true=y_train, cv=cv, name=name)


# imbalanced data does not hurt knns
def knn(x_train, y_train, cv, n_neighbors=20, weights="distance", name="KNearestNeighbours", **kwargs):
    """ Support Vector Machine with Radial Basis functions as kernel.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        n_neighbors (int): Number of neighbors to consider
        weights (str): either "distance" or "uniform". uniform is a simple majority vote.
            distance means that the nieghbours are weighted according to their distances.
        name (str): Name/Description for the model
    Returns:
        dict: contains results of models
    """
    knn = KNeighborsClassifier(n_neighbors = n_neighbors,
                               weights = "distance")
    return calculate_metrics_cv(model=knn, X=x_train, y_true=y_train, cv=cv, name=name)

# specifically for imbalanced data
# https://imbalanced-learn.org/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html
def ada_boost(x_train, y_train, cv, n_estimators=100, sampling_strategy="not_majority", name="AdaBoost", **kwargs):
    """Bags AdaBoost learners which are trained on balanced bootstrap samples.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        n_estimators (int): number of boosted trees to consider
        sampling_strategy (str): "all", "not_majority", "minority" and more. See docu of classifer for more details.
        name (str): Name/Description for the model
    Returns:
        dict: contains results of models
    """
    eec = EasyEnsembleClassifier(n_estimators=n_estimators,
                                 sampling_strategy=sampling_strategy,
                                 random_state=42)
    return calculate_metrics_cv(model=eec, X=x_train, y_true=y_train, cv=cv, name=name)
