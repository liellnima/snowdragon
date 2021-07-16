import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_handling.data_parameters import MODEL_COLORS, SNOW_TYPES_SELECTION


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


def main():
    # load dataframe with performance data
    all_scores = pd.read_csv("../data/all_scores02.csv")
    label_acc = pd.read_csv("../data/all_acc_per_label02.csv")
    label_prec = pd.read_csv("../data/all_prec_per_label02.csv")

    # visualize the original data
    plot_model_comparison(
        label_acc, plot_rare=False, save_path="../plots/evaluation/model_comparison.png"
    )


if __name__ == "__main__":
    main()
