# here all the different metrics are saved as scoreres
from sklearn.metrics import make_scorer, balanced_accuracy_score, recall_score
from sklearn.metrics import precision_score, roc_auc_score, log_loss, confusion_matrix
import numpy as np

# Wrapper functions for metrics, see scikit learn docu for more info
# important: average=None means that the metric is calculated per each label
# (not avilable for multiclass case in roc auc or log_loss)
def balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def recall(y_true, y_pred, average="weighted", labels=None):
    return recall_score(y_true, y_pred, average=average, labels=labels, zero_division=0)

def precision(y_true, y_pred, average="weighted", labels=None):
    return precision_score(y_true, y_pred, average=average, labels=labels, zero_division=0)

def roc_auc(y_true, y_pred, labels=None):
    return roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr", labels=labels)#, labels=np.unique(y_true)) # TODO check if ovo makes more sense

def my_log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)

def my_confusion_matrix(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)

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
def calculate_metrics_raw(y_trues, y_preds, metrics=METRICS, cv=True, name=None, annot="train"):
    """ Calculates metrics when already the list of the observed and predicted target values is given. E.g. from a manual cross validation.
    Paramters:
        y_preds (list): List of lists (crossvalidation!) with predicted target values
        y_trues (list): List of listst (crossvalidation!) with true or observed target values
        metrics (dict): Dictionary were the keys describe the metrics and the values are callable functions containing y_true and y_pred as parameters.
        cv (bool): Indicates if y_preds and y_trues come from a crossvalidation and are nested lists. If not, they are not nested lists!
        name (String): Name of the model evaluated
        annot (String): indicates if we speak about train or test or validation data
    Returns:
        dict: dictionary with different scores
    """
    # convert metrics to list
    funcs = list(metrics.values())
    metric_names = list(metrics.keys())
    if annot is not None: annot = annot + "_"
    scores = {}
    if name is not None:
        scores["model"] = name
    # iterate through a list of metric functions and add the lists of results to scores
    if cv:
        for func, name in zip(funcs, metric_names):
            scores[annot + name] = np.asarray([func(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)])
    else:
        for func, name in zip(funcs, metric_names):
            scores[annot + name] = func(y_trues, y_preds)

    return scores

# helper function to calculate all metrics per label (not cv compatible)
def calculate_metrics_per_label(y_trues, y_preds, name=None, annot="train", labels_order=None):
    """ Calculates metrics when already the list of the observed and predicted target values is given.
    Does not work for crossvalidation, only during evaluation. Does return the confusion matrix,
    accuracy, precision and recall per each label. No probability based metrics.
    The order of the labels is always sorted ascending (from 0 to inf) if not indicated otherwise.
    Paramters:
        y_trues: List with true or observed target values
        y_preds: List with predicted target values
        y_prob_preds (list): List with predicted probabilities of target values
        name (String): Name of the model evaluated
        annot (String): indicates if we speak about train or test or validation data
        labels_order (list): order of the labels. If None the order will be ascending.
    Returns:
        dict: dictionary with different scores
    """
    scores = {}
    if annot is not None: annot = annot + "_"
    if name is not None: scores["model"] = name
    if labels_order is None: labels_order = np.sort(np.unique(y_trues))

    # add precision, recall, roc_auc and confusion matrix
    scores[annot + "confusion_matrix"] = my_confusion_matrix(y_trues, y_preds, labels=labels_order)
    # add accuracy per label (diagonal of matrix)
    cm = scores[annot + "confusion_matrix"]
    # normalize confusion matrix and take the diagonal -> acc per label
    accuracies = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).diagonal()
    scores[annot + "accuracy"] = accuracies
    scores[annot + "recall"] = recall(y_trues, y_preds, average=None, labels=labels_order)
    scores[annot + "precision"] = precision(y_trues, y_preds, average=None, labels=labels_order)

    return scores
