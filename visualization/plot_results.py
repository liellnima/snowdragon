import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_handling.data_parameters import MODEL_COLORS, SNOW_TYPES_SELECTION, ANTI_LABELS
from models.helper_funcs import load_results


def plot_model_comparison(performances, plot_rare=False, save_path=""):
    """Visualize the model accuracies for all classes

    Parameters:
        performances (df.DataFrame): Accuracies of the different models
        on individual classes
    """

    # Rearrange the DataFrame
    performances_rs = pd.DataFrame(columns={"class", "accuracy", "model"})
    for row in performances.index:
        for c in SNOW_TYPES_SELECTION:
            if c != "model":
                performances_rs = performances_rs.append(
                    {
                        "class": c,
                        "accuracy": performances.loc[row][c],
                        "model": performances.loc[row]["model"],
                    },
                    ignore_index=True,
                )

    if plot_rare:
        data = performances_rs
    else:
        data = performances_rs[performances_rs["class"] != "rare"]

    plt.figure(figsize=(15, 7))
    ax = sns.pointplot(
        data=data,
        x="class",
        y="accuracy",
        hue="model",
        palette=MODEL_COLORS,
        scale=2,
    )
    plt.setp(ax.lines, alpha=0.7)
    plt.legend(loc=0, bbox_to_anchor=(1.0, 1), fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("")
    plt.ylabel("Accuracy", fontsize=20)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if len(save_path) > 0:
        # TODO: Look up dpi and format requirements of journal
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()

def plot_confusion_matrix(confusion_matrices, label_orders, names, save_file=None):
    """ Plot confusion matrix with relative prediction frequencies per label as heatmap.
    Parameters:
        confusion_matrices (list of nested list): for each model:
            2d nested list with frequencies
        label_orders (list of lists): for each model:
            list of tags or labels that should be used for the plot.
            Must be in the same order like the label order of the confusion matrix.
        names (list of strs): Names of the model for the plot
        save_file (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

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
        box_annots = [f"{v3}\n\n{v1}{v2}".strip() for v1, v2, v3 in zip(counts, percentages, str_accs)]
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
                        annot_kws={"fontsize":6}, ax=ax)
        # change font size of cbar axis
        #g.figure.axes[-1].yaxis.label.set_size(14)
        ax.set_title("Model {}, {}".format(name, stats_text))
        # if i%2 ...
        if i == 0:
            ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.subplots_adjust(hspace=0.3)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()

# for full model names include dictionary!
def prepare_vis_data(eval_dir):
    """ Preparing the raw data from evaluation for visualization.

    Parameters:
        eval_dir (str): String path to the directory where the evaluation results
            of the different models are stored

    Returns:
        triple: (names, confusion_matrices_list, label_orders_list)
    """
    cf_matrices = []
    label_orders = []
    names = []

    # get all model subdirs
    model_subdirs = next(os.walk(eval_dir))[1]

    for model in model_subdirs:
        metric = load_results(eval_dir+"/"+model+"/eval_raw_data/metrics_per_label.pickle")
        order = load_results(eval_dir+"/"+model+"/eval_raw_data/labels_order.pickle")
        cf_matrices.append(metric["eval_confusion_matrix"])
        label_orders.append(order)
        names.append(model)

    return names, cf_matrices, label_orders

def main():
    # load dataframe with performance data
    all_scores = pd.read_csv("data/scores/all_scores.csv")#("../data/all_scores02.csv")
    label_acc = pd.read_csv("data/scores/acc_labels.csv")#("../data/all_acc_per_label02.csv")
    label_prec = pd.read_csv("data/scores/prec_labels.csv")#("../data/all_prec_per_label02.csv")

    # retrieve and summarize the data for the confusion matrices and the roc curves
    names, cf_matrices, label_orders = prepare_vis_data("plots/evaluation")

    plot_confusion_matrix(cf_matrices, label_orders, names)

    exit(0)

    # visualize the accuracies of the different models
    plot_model_comparison(
        label_acc, plot_rare=False, save_path="../plots/evaluation/model_comparison.png"
    )

    # plot confusion matrices

    # plot roc curves

if __name__ == "__main__":
    main()
