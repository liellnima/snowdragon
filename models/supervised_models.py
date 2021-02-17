from models.cv_handler import calculate_metrics_cv

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier

# TODO make name a parameter and return_train_score as well

def random_forest(x_train, y_train, cv, name="RandomForest"):
    """ Random Forest.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        name (str): Name/Description for the model
    Returns:
        dict: contains results of models
    """
    rf = RandomForestClassifier(n_estimators=10,
                                criterion = "entropy",
                                bootstrap = True,
                                max_samples = 0.6,     # 60 % of the training data (None: all)
                                max_features = "sqrt", # uses sqrt(num_features) features
                                class_weight = "balanced", # balanced_subsample computes weights based on bootstrap sample
                                random_state = 42)
    return calculate_metrics_cv(model=rf, X=x_train, y_true=y_train, cv=cv, name=name)

def svm(x_train, y_train, cv, gamma="auto", name="SupportVectorMachine"):
    """ Support Vector Machine with Radial Basis functions as kernel.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        gamma (num or Str): gamma value for svm
        name (str): Name/Description for the model
    Returns:
        dict: contains results of models
    """
    svm = SVC(decision_function_shape = "ovr",
              kernel = "rbf",
              gamma = gamma,
              class_weight = "balanced",
              random_state = 24)
    return calculate_metrics_cv(model=svm, X=x_train, y_true=y_train, cv=cv, name=name)


# imbalanced data does not hurt knns
def knn(x_train, y_train, cv, n_neighbors, name="KNearestNeighbours"):
    """ Support Vector Machine with Radial Basis functions as kernel.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        n_neighbors: Number of neighbors to consider
        name (str): Name/Description for the model
    Returns:
        dict: contains results of models
    """
    knn = KNeighborsClassifier(n_neighbors = n_neighbors,
                               weights = "distance")
    return calculate_metrics_cv(model=knn, X=x_train, y_true=y_train, cv=cv, name=name)

# specifically for imbalanced data
# https://imbalanced-learn.org/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html
def ada_boost(x_train, y_train, cv, name="AdaBoost"):
    """Bags AdaBoost learners which are trained on balanced bootstrap samples.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        name (str): Name/Description for the model
    Returns:
        dict: contains results of models
    """
    eec = EasyEnsembleClassifier(n_estimators=100,
                                 sampling_strategy="all",
                                 random_state=42)
    return calculate_metrics_cv(model=eec, X=x_train, y_true=y_train, cv=cv, name=name)
