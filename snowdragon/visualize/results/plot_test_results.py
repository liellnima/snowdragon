import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_curve, auc

from snowdragon import OUTPUT_DIR

def plot_test_confusion_matrix(
        confusion_matrix: list, 
        labels: list, 
        name: str = "", 
        file_name= OUTPUT_DIR / "plots_data" / "confusion_matrix.png",
    ):
    """ Plot confusion matrix with relative prediction frequencies per label as heatmap.
    Parameters:
        confusion_matrix (nested list): 2d nested list with frequencies
        labels (list): list of tags or labels that should be used for the plot.
            Must be in the same order like the label order of the confusion matrix.
        name (str): Name of the model for the plot
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    # "accuracy per label" so to say (diagonal at least is acc)
    bal_accs = [[cell/sum(row) for cell in row] for row in confusion_matrix]

    # absolute counts and percentages
    counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    sum_count = sum(confusion_matrix.flatten())
    str_accs = [r"$ \bf {0:0.3f} $".format(value) for row in bal_accs for value in row]
    percentages = [" ({0:.2f})".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
    percentages = [" ({0:.1%})".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
    # annotations inside the boxes
    box_annots = [f"{v3}\n\n{v1}{v2}".strip() for v1, v2, v3 in zip(counts, percentages, str_accs)]
    box_annots = np.asarray(box_annots).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])

    # Total accuracy: summed up diagonal / tot obs
    accuracy  = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
    stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)

    # plot the matrix
    g = sns.heatmap(bal_accs, annot=box_annots, fmt="", cmap="Blues", cbar=True,
                    xticklabels=labels, yticklabels=labels, vmin=0, vmax=1,
                    cbar_kws={"label": "\nPrediction Frequency per Label"},
                    annot_kws={"fontsize":6})
    # change font size of cbar axis
    #g.figure.axes[-1].yaxis.label.set_size(14)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label" + stats_text)
    plt.title("Confusion Matrix of {} Model \n".format(name))
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, dpi=200)
        plt.close()

def plot_test_roc_curve(
        y_trues: pd.DataFrame, # unsure (documented too late)
        y_prob_preds: pd.DataFrame, # unsure (documented too late)
        roc_labels: list, # unsure (documented too late)
        anti_labels: dict,
        colors: dict,
        name: str = "", 
        file_name: Path = OUTPUT_DIR / "plots_data" / "roc_curve.png",
    ):
    """ Plotting ROC curves for all labels of a multiclass classification problem.
    Inspired from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    Parameters:
        y_trues (pd.DataFrame): targets
        y_prob_preds (pd.DataFrame): predicted values for targets (different probabilities for each grain type)
        roc_labels (list): the grain types considered for plotting the roc curve
        anti_labels (dict): Label dictionary from configs, inversed. Matches the number identifier (int) to the string describing the grain: <int: str> 
        colors (dict): Color dictionary from configs. Matches a grain number (int) to a color: < grain(int): color(str)>
        name (str): name of the model
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    n_classes = len(roc_labels)

    # Compute ROC curve and ROC area for each class
    false_pos_rate = dict()
    true_pos_rate = dict()
    roc_auc = dict()
    # calculate the one-hot-encoding for y_trues
    y_trues_dummies = pd.get_dummies(y_trues, drop_first=False).values
    for i in range(n_classes):
        false_pos_rate[i], true_pos_rate[i], _ = roc_curve(y_trues_dummies[:, i], y_prob_preds[:, i])
        roc_auc[i] = auc(false_pos_rate[i], true_pos_rate[i])

    # Compute micro-average ROC curve and ROC area
    false_pos_rate["micro"], true_pos_rate["micro"], _ = roc_curve(y_trues_dummies.ravel(), y_prob_preds.ravel())
    roc_auc["micro"] = auc(false_pos_rate["micro"], true_pos_rate["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([false_pos_rate[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, false_pos_rate[i], true_pos_rate[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    false_pos_rate["macro"] = all_fpr
    true_pos_rate["macro"] = mean_tpr
    roc_auc["macro"] = auc(false_pos_rate["macro"], true_pos_rate["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(false_pos_rate["micro"], true_pos_rate["micro"],
             label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
             color="deeppink", linestyle=':', linewidth=4)

    plt.plot(false_pos_rate["macro"], true_pos_rate["macro"],
             label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
             color="blueviolet", linestyle=':', linewidth=4)

    my_colors = [colors[roc_label] for roc_label in roc_labels]
    for i, color, roc_label in zip(range(n_classes), my_colors, roc_labels):
        plt.plot(false_pos_rate[i], true_pos_rate[i], color=color, lw=2,
                 label="ROC curve of class {0} (area = {1:0.2f})".format(anti_labels[roc_label], roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristics for Model {}\n".format(name))
    plt.legend(loc="lower right", prop={'size': 7})
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()