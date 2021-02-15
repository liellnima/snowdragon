# here all the different metrics are saved as scoreres
from sklearn.metrics import make_scorer, balanced_accuracy_score, recall_score, precision_score, roc_auc_score, log_loss
import numpy as np

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
