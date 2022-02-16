import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from data_handling.data_parameters import MODEL_COLORS, SNOW_TYPES_SELECTION, COLORS
from data_handling.data_parameters import ANTI_LABELS, ANTI_LABELS_LONG, LABELS, MODEL_LABELS
from models.helper_funcs import load_results


def plot_roc_auc(y_trues, y_prob_preds, labels, names, legend=True, file_name=None):
    """ Plotting ROC curves for all labels of a multiclass classification problem.
    Inspired from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    Parameters:
        y_trues (list<list>): True y values for each model (list of lists)
        y_prob_preds (list<list>): Predicted probabilities for each model (list of lists)
        labels (list<list>): Labels for each model (list of lists)
        names (list<str>): Simple list with model names
        legend (bool): If the legend should be plot in a separate panel
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    # set up figure
    plt.rcParams["figure.dpi"] = 500
    number_rows = int(math.ceil(len(y_trues) / 2))
    height = number_rows * (6 + 2/3) # 3 -> 20, 1 -> 6.66
    fig, axs = plt.subplots(nrows=number_rows, ncols=2, sharex=False, sharey=True, figsize=(15, height), dpi=500)
    # remove /n from ANTI_LABELS_LONG
    anti_labels_super_long = {key: value.replace("\n", ' ') for key, value in ANTI_LABELS_LONG.items()}

    handels = []
    legend_labels = []

    # iterate through models
    for i, ax in enumerate(axs.flat):
        if i >= len(names):
            if legend:
                ax.axis("off")
                ax.set_title("ROC Curves Legend", fontsize=20)
                ax.legend(handels, legend_labels, loc="center left", prop={"size": 17}, frameon=True, borderpad=1.5, labelspacing=0.8)
                break
            else:
                break
        y_true = y_trues[i]
        y_pred = y_prob_preds[i]
        label = labels[i]
        name = names[i]

        # compute rates needed for plotting
        false_pos_rate, true_pos_rate, roc_auc = compute_auc_rates(y_true, y_pred, label)

        # Plot all ROC curves
        ax.plot(false_pos_rate["micro"], true_pos_rate["micro"],
                 label="Micro-average",
                 color="deeppink", linestyle=':', linewidth=4)

        ax.plot(false_pos_rate["macro"], true_pos_rate["macro"],
                 label="Macro-average",
                 color="blueviolet", linestyle=':', linewidth=4)

        colors = [COLORS[l] for l in label]
        for i, color, l in zip(range(len(label)), colors, label):
            ax.plot(false_pos_rate[i], true_pos_rate[i], color=color, lw=2,
                     label="Class {}".format(anti_labels_super_long[l]))

        ax.plot([0, 1], [0, 1], "k--", lw=2)
        ax.set_title("{0}, Micro-avg area: {1:0.2f}".format(MODEL_LABELS[name], roc_auc["micro"]), fontsize=20)
        handels, legend_labels = ax.get_legend_handles_labels()

    # shared figure stuff
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #plt.legend(loc="lower right", prop={'size': 7})
    fig.text(0.5, 0.05, "False Positive Rate", ha="center", fontsize=25)
    fig.text(0.07, 0.5, "True Positive Rate", va="center", rotation="vertical", fontsize=25)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def compute_auc_rates(y_trues, y_prob_preds, labels):
    """ Computes necessary statistics for roc auc curve
    Parameters:
        y_trues (list): true y values
        y_prob_preds (list): probabilities of predicted y
        labels (list): labels used
    Returns:
        3-tuple <dict>: three dictionaries with false positive rate,
            true positive rate and roc auc scores
    """
    n_classes = len(labels)

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

    return false_pos_rate, true_pos_rate, roc_auc


def plot_model_comparison(performances, plot_rare=False, file_name="", metric_name="accuracy"):
    """Visualize the model accuracies for all classes

    Parameters:
        performances (df.DataFrame): Accuracies of the different models
            on individual classes
        plot_rate (bool): Default=False. If the class rare should be included or not.
        file_name (str): If empty, the plot is shown. Else the plot is saved at the given path.
        metric_name (str): Default is "accuracy". Name of the performance metric
    """
    # Rearrange the DataFrame
    performances_rs = pd.DataFrame(columns={"class", metric_name, "model"})
    for row in performances.index:
        for c in SNOW_TYPES_SELECTION:
            if c != "model":
                performances_rs = performances_rs.append(
                    {
                        "class": c,
                        metric_name: performances.loc[row][c],
                        "model": performances.loc[row]["model"],
                    },
                    ignore_index=True,
                )

    if plot_rare:
        data = performances_rs
    else:
        data = performances_rs[performances_rs["class"] != "rare"]

    # TODO: individual alpha values, the higher the performance the higher the alpha
    plt.figure(figsize=(15, 7))
    ax = sns.pointplot(
        data=data,
        x="class",
        y=metric_name,
        hue="model",
        palette=MODEL_COLORS,
        scale=2
    )
    plt.setp(ax.lines, alpha=0.6, linestyle="-")

    # set up legend
    handles, labels = ax.get_legend_handles_labels()
    changed_labels = []
    # add balanced accuracy to legend
    for label in labels:
        model_data = data[data["model"]==label]
        overall_acc = model_data[metric_name].sum()/ len(model_data)
        changed_labels.append("%s (%4.2f)" % (label, overall_acc))

    plt.legend(loc=0, bbox_to_anchor=(1.0, 1),
        title="Models (balanced {})".format(metric_name),
        title_fontsize=20, fontsize = 15,
        labelspacing=0.75,
        labels=changed_labels, handles=handles)

    # set up x tick labels and the other labels
    long_xtick_labels = []
    xtick_locs, xtick_labels = plt.xticks()
    for xlabel in xtick_labels:
        xlabel = xlabel.get_text()
        long_xtick_labels.append(ANTI_LABELS_LONG[LABELS[xlabel]])
    plt.xticks(fontsize=15, ticks=xtick_locs, labels=long_xtick_labels)
    plt.yticks(fontsize=20)
    plt.xlabel("Snow Grain Types", fontsize=20)
    plt.ylabel(metric_name.title(), fontsize=20)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if len(file_name) > 0:
        # TODO: Look up dpi and format requirements of journal
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    else:
        plt.show()

def plot_confusion_matrix(confusion_matrices, label_orders, names, file_name=None):
    """ Plot confusion matrix with relative prediction frequencies per label as heatmap.
    Parameters:
        confusion_matrices (list of nested list): for each model:
            2d nested list with frequencies
        label_orders (list of lists): for each model:
            list of tags or labels that should be used for the plot.
            Must be in the same order like the label order of the confusion matrix.
        names (list of strs): Names of the model for the plot
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    plt.rcParams['figure.dpi'] = 500
    number_rows = int(math.ceil(len(confusion_matrices) / 2))
    height = number_rows * (6 + 2/3) # 3 -> 20, 1 -> 6.66
    fig, axs = plt.subplots(nrows=number_rows, ncols=2, sharex=True, sharey=True, figsize=(15, height), dpi=500)
    cbar_ax = fig.add_axes([.93, .3, .05, .4])

    for i, ax in enumerate(axs.flat):
        if i >= len(names): break
        confusion_matrix = confusion_matrices[i]
        labels = [ANTI_LABELS[label] for label in label_orders[i]]
        name = names[i]

        # "accuracy per label" so to say (diagonal at least is acc)
        bal_accs = [[cell/sum(row) for cell in row] for row in confusion_matrix]

        # absolute counts and percentages
        counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
        sum_count = sum(confusion_matrix.flatten())
        str_accs = [r"$ \bf {0:0.3f} $".format(value) for row in bal_accs for value in row]
        percentages = [" ({0:.2f})".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
        percentages = [" ({0:.1%})".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
        # annotations inside the boxes
        #box_annots = [f"{v3}\n\n{v1}{v2}".strip() for v1, v2, v3 in zip(counts, percentages, str_accs)]
        box_annots = [f"{v}".strip() for v in str_accs]
        box_annots = np.asarray(box_annots).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])

        # Total accuracy: summed up diagonal / tot obs
        accuracy  = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
        stats_text = "Accuracy={:0.3f}".format(accuracy)

        # plot the matrix
        g = sns.heatmap(bal_accs, annot=box_annots, fmt="", cmap="Blues",
                        xticklabels=labels, yticklabels=labels,
                        cbar=i == 0, vmin=0, vmax=1,
                        cbar_ax=None if i else cbar_ax,
                        cbar_kws={"label": "\nPrediction Frequency per Label"},
                        annot_kws={"fontsize":10}, ax=ax)
        # change font size of cbar axis and ticks
        g.figure.axes[-1].yaxis.label.set_size(20)
        if not i:
            cbar = g.collections[0].colorbar
            cbar.ax.tick_params(labelsize=15)
        g.set_yticklabels(labels, size = 15, va="center")
        g.set_xticklabels(labels, size = 15, ha="center")
        ax.set_title("{}, {}".format(MODEL_LABELS[name], stats_text), fontsize=20)

    #fig.supxlabel("Predicted Label")
    #fig.supylabel("True Label")
    text_height = 0.07 if number_rows > 1 else 0.01
    fig.text(0.5, text_height, "Predicted Label", ha="center", fontsize=25)
    fig.text(0.07, 0.5, "True Label", va="center", rotation="vertical", fontsize=25)
    plt.subplots_adjust(hspace=0.2)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0.8)
        plt.close()

# for full model names include dictionary!
def prepare_evaluation_data(eval_dir):
    """ Preparing the raw data from evaluation for visualization.

    Parameters:
        eval_dir (str): String path to the directory where the evaluation results
            of the different models are stored

    Returns:
        n-tuple of lists: (names, confusion_matrices_list, label_orders_list, y_true, y_pred_probs)
    """
    cf_matrices = []
    label_orders = []
    names = []
    y_trues = []
    y_pred_probs = []

    # get all model subdirs
    model_subdirs = next(os.walk(eval_dir))[1]

    for model in model_subdirs:
        metric = load_results(eval_dir+"/"+model+"/eval_raw_data/metrics_per_label.pickle")
        order = load_results(eval_dir+"/"+model+"/eval_raw_data/labels_order.pickle")
        y_true = load_results(eval_dir+"/"+model+"/eval_raw_data/y_test.pickle")
        y_pred_prob = load_results(eval_dir+"/"+model+"/eval_raw_data/y_pred_prob.pickle")
        y_trues.append(y_true)
        y_pred_probs.append(y_pred_prob)
        cf_matrices.append(metric["eval_confusion_matrix"])
        label_orders.append(order)
        names.append(model)

    return names, cf_matrices, label_orders, y_trues, y_pred_probs
