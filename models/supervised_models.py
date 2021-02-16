from models.cv_handler import calculate_metrics_cv

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier

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

# specifically for imbalanced data
# https://imbalanced-learn.org/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html
def ada_boost(x_train, y_train, cv):
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
