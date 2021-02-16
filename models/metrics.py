# here all the different metrics are saved as scoreres
from sklearn.metrics import make_scorer, balanced_accuracy_score, recall_score, precision_score, roc_auc_score, log_loss
import numpy as np

# Wrapper functions for metrics
def balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="weighted", zero_division=0)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="weighted", zero_division=0)

def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")#, labels=np.unique(y_true)) # TODO check if ovo makes more sense

def my_log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)

# Constant dictionaries collecting the fixed functions.
SCORERS = {"balanced_accuracy": make_scorer(balanced_accuracy),
           "recall": make_scorer(recall),
           "precision": make_scorer(precision)}
# make_scorere wrapper is necessary for cross_validate function from scikit learn
METRICS = {"balanced_accuracy": balanced_accuracy,
           "recall": recall,
           "precision": precision}
# for those metrics I use my own cross validation, no make_scorer is necessary for this!
METRICS_PROB = {"roc_auc": roc_auc,
                "log_loss": my_log_loss}

# helper function to calculate some of the metrics
def calculate_metrics_raw(y_trues, y_preds, metrics=METRICS, name=None, annot="train"):
    """ Calculates metrics when already the list of the observed and predicted target values is given. E.g. from a manual cross validation.
    Paramters:
        y_preds (list): List of lists (crossvalidation!) with predicted target values
        y_trues (list): List of listst (crossvalidation!) with true or observed target values
        metrics (dict): Dictionary were the keys describe the metrics and the values are callable functions containing y_true and y_pred as parameters.
        name (String): Name of the model evaluated
        annot (String): indicates if we speak about train or test or validation data
    Returns:
        dict: dictionary with different scores
    """
    # TODO convert metrics to list
    funcs = list(metrics.values())
    metric_names = list(metrics.keys())
    if annot is not None: annot = annot + "_"
    scores = {}
    if name is not None:
        scores["model"] = name
    # iterate through a list of metric functions and add the lists of results to scores
    for func, name in zip(funcs, metric_names):
        scores[annot + name] = np.asarray([func(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)])

    return scores
