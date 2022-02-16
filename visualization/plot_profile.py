import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from data_handling.data_preprocessing import idx_to_int
from data_handling.data_parameters import COLORS, ANTI_LABELS, ANTI_LABELS_LONG


# TODO
# make function to compare predicted profiles where we dont have the ground truth!
# adapt compare_plot accordingly --> essentially: do this with ini files only!

def main():
    """ This function exists only for testing purposes.
    """
    # read in example data and extract prediction and true data
    with open("visualization/storage/smp_Random Forest_S31H0234.pickle", "rb") as handle:
        rf_true, rf_pred = pickle.load(handle)
    with open("visualization/storage/smp_Baseline_S31H0234.pickle", "rb") as handle:
        base_true, base_pred = pickle.load(handle)

    smp_preds = [base_pred, rf_pred]
    smp_true = base_true
    smp_name = "S31H0488"
    smp_name = "S31H0234"
    model_names = ["Baseline", "Random Forest"]
    compare_plot(smp_true, smp_preds, smp_name, model_names, title="", grid=True, file_name="output/evaluation/smp_compare_example.png")

def compare_plot(smp_true, smp_preds, smp_name, model_names, title=None, grid=True, file_name="output/plots_data/smp_compare.png"):
    """
    """
    if isinstance(smp_name, str):
        smp_wanted = idx_to_int(smp_name)
    else:
        smp_wanted = smp_name

    #differences = smp_true["label"] != smp_pred["label"]

    smps = [smp_true, smp_preds[0], smp_preds[1]]

    fig, axs = plt.subplots(3, sharex=True, sharey=True, dpi=300)
    first_ax = True
    line_handles = []
    for ax, smp in zip(axs, smps):
        # calculate differences between true and pred
        differences = smp_true["label"] != smp["label"]

        if first_ax:
            alpha = 0.5
            # plot the ground truth measurement
            #["Force Signal", "Wrong Classification"]
            signalplot = sns.lineplot(smp["distance"], smp["mean_force"], ax=ax)
            line_handles.append(signalplot)
            first_ax = False
        else:
            alpha = 0.2
            # plot the differences
            diff = differences.astype(int) * (max(smp["mean_force"]) / 2) # plot line in the middle
            diff = diff.replace({0:np.inf}) # cheating a bit: seaborn wont plot inf values
            # - 0.5 is only because the plot lines have a small off-set (not numerically though)
            lineplot = sns.lineplot(smp["distance"]-0.5, diff, ax=ax, linewidth=4, color='r')
            if len(line_handles) < 2:
                line_handles.append(lineplot)
            # plot single differences
            single_diffs = list(diff)
            diff_points = []
            for i, d in enumerate(single_diffs):
                if i > 1 and i < len(single_diffs)-1:
                    if (single_diffs[i-1] == np.inf) and (single_diffs[i+1] == np.inf) and (single_diffs[i] != np.inf):
                        diff_points.append(i)
            points_indices = smp["distance"].index[diff_points]
            points_heights = [(max(smp["mean_force"]) / 2)] * len(points_indices)
            scatterplot = plt.scatter(smp["distance"][points_indices] - 0.5, points_heights, marker='s', linewidth=0, color='r')

        last_label_num = 1
        last_distance = -1
        # going through labels and distance
        for label_num, distance in zip(smp["label"], smp["distance"]):
            if (label_num != last_label_num):
                # assign new background for each label
                background = ax.axvspan(last_distance, distance-1, color=COLORS[last_label_num], alpha=alpha)
                last_label_num = label_num
                last_distance = distance-1

            if distance == smp.iloc[len(smp)-1]["distance"]:
                ax.axvspan(last_distance, distance, color=COLORS[label_num], alpha=alpha)
        ax.set_xlim(0, len(smp)-1)
        ax.set_ylim(0)
        ax.set(ylabel=None)

    names = ["Ground Truth"] + model_names
    for name, ax in zip(names, axs):
        ax.text(0.0, 1.05, name, fontweight="bold", fontsize=8.5, transform=ax.transAxes)

    # Snow Grain Lagend
    list_all_labels = [label for smp in smps for label in [*smp["label"].unique()]]
    all_labels = list(set(list_all_labels))
    anti_colors = {ANTI_LABELS_LONG[key] : value for key, value in COLORS.items() if key in all_labels}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', alpha=0.5) for color in anti_colors.values()]
    legend1 = plt.figlegend(markers, anti_colors.keys(), numpoints=1, loc=7, title="Snow Grain Types")# borderaxespad=0
    fig.add_artist(legend1)

    # Signal and Mislassification lines legend
    blue_line = mlines.Line2D([], [], label="Force Signal")
    red_line = mlines.Line2D([], [], color="red", linewidth=4, label="Misclassified")
    legend2 = plt.legend(handles=[blue_line, red_line], bbox_to_anchor=(1.04, 0.5))
    fig.add_artist(legend2)

    if title is None:
        plt.suptitle("Observed and Predicted SMP Profile {}".format(smp_name))
    else:
        plt.suptitle(title)
    fig.text(0.015,0.5, "Mean Force [N]", ha="center", va="center", rotation=90)
    plt.xlabel("Snow Depth [mm]")
    plt.tight_layout()
    fig.subplots_adjust(right=0.72)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()
    pass

def smp_unlabelled(smp, smp_name, file_name="output/plots_data/smp_unlabelled.png"):
    """ Plots a SMP profile without labels.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        smp_name (String or float/int): Name of the wished smp profile or
            alternatively its converted index number
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    if isinstance(smp_name, str):
        smp_wanted = idx_to_int(smp_name)
    else:
        smp_wanted = smp_name
    smp_profile = smp[smp["smp_idx"] == smp_wanted]
    ax = sns.lineplot(smp_profile["distance"], smp_profile["mean_force"])
    plt.title("Unlabelled SMP Profile {}".format(smp_name))
    ax.set_xlabel("Snow Depth [mm]")
    ax.set_ylabel("Mean Force [N]")
    ax.set_ylim(0)
    ax.set_xlim(0, len(smp_profile)-1)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def smp_labelled(smp, smp_name, title=None, file_name="output/plots_data/smp_labelled.pngs"):
    """ Plots a SMP profile with labels.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        smp_name (String or float/int): Name of the wished smp profile or
            alternatively its converted index number
        title (str): if None, a simple headline for the plot is used.
            Please specify with string.
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    # SHOW THE SAME PROFILE WITH LABELS
    if isinstance(smp_name, str):
        smp_wanted = idx_to_int(smp_name)
    else:
        smp_wanted = smp_name

    smp_profile = smp[smp["smp_idx"] == smp_wanted]
    ax = sns.lineplot(smp_profile["distance"], smp_profile["mean_force"])
    if title is None:
        plt.title("{} SMP Profile Distance (1mm layers) and Force\n".format(smp_name))
    else:
        plt.title(title)
    last_label_num = 1
    last_distance = -1
    # going through labels and distance
    for label_num, distance in zip(smp_profile["label"], smp_profile["distance"]):
        if (label_num != last_label_num):
            # assign new background for each label
            background = ax.axvspan(last_distance, distance-1, color=COLORS[last_label_num], alpha=0.5)
            last_label_num = label_num
            last_distance = distance-1

        if distance == smp_profile.iloc[len(smp_profile)-1]["distance"]:
            ax.axvspan(last_distance, distance, color=COLORS[label_num], alpha=0.5)


    anti_colors = {ANTI_LABELS[key] : value for key, value in COLORS.items() if key in smp_profile["label"].unique()}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', alpha=0.5) for color in anti_colors.values()]
    plt.legend(markers, anti_colors.keys(), numpoints=1, loc="lower right")
    ax.set_xlabel("Snow Depth [mm]")
    ax.set_ylabel("Mean Force [N]")
    ax.set_xlim(0, len(smp_profile)-1)
    ax.set_ylim(0)
    plt.tight_layout()
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def smp_pair(smp_true, smp_pred, smp_name, title=None, grid=True, file_name="output/plots_data/smp_pair.png"):
    """ Visualizes the prediced and the observed smp profile in one plot.
    The observation is a bar above/inside the predicted smp profile.
    Parameters:
        smp_true (pd.DataFrame): Only one SMP profile -the observed one-, which is already filtered out.
        smp_pred (pd.DataFrame): Only one SMP profile -the predicted one-, which is already filtered out.
        smp_name (num or str): Name of the SMP profile that is observed and predicted.
        title (str): Title of the plot.
        grid (bool): If a grid should be plotted over the plot.
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    if isinstance(smp_name, str):
        smp_wanted = idx_to_int(smp_name)
    else:
        smp_wanted = smp_name

    # create two subplots with 1:10 ratio
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 10]})

    # plot the observed smp profile by plotting only background for each label
    last_label_num = 1
    last_distance = -1
    for label_num, distance in zip(smp_true["label"], smp_true["distance"]):
        if (label_num != last_label_num):
            background = axs[0].axvspan(last_distance, distance-1, color=COLORS[last_label_num], alpha=0.5)
            last_label_num = label_num
            last_distance = distance-1
        if distance == smp_true.iloc[len(smp_true)-1]["distance"]:
            axs[0].axvspan(last_distance, distance, color=COLORS[label_num], alpha=0.5)
    axs[0].set_xlim(0, len(smp_pred))
    axs[0].set_yticks([])
    plt.text(0.01, 0.5, "Ground Truth", fontweight="bold", fontsize=8.5, transform=axs[0].transAxes)

    # plot the predicted smp profile - with line and background colors
    axs[1] = sns.lineplot(smp_pred["distance"], smp_pred["mean_force"], ax=axs[1])
    last_label_num = 1
    last_distance = -1
    for label_num, distance in zip(smp_pred["label"], smp_pred["distance"]):
        if (label_num != last_label_num):
            background = axs[1].axvspan(last_distance, distance-1, color=COLORS[last_label_num], alpha=0.5)
            last_label_num = label_num
            last_distance = distance-1
        if distance == smp_pred.iloc[len(smp_pred)-1]["distance"]:
            axs[1].axvspan(last_distance, distance, color=COLORS[label_num], alpha=0.5)
    axs[1].set_xlim(0, len(smp_pred)-1)
    axs[1].set_ylim(0)
    plt.text(0.01, 0.95, "Prediction", fontweight="bold", fontsize=8.5, transform=axs[1].transAxes)

    # set legend, and axis labels for prediction
    all_labels = list(set([*smp_true["label"].unique(), *smp_pred["label"].unique()]))
    anti_colors = {ANTI_LABELS[key] : value for key, value in COLORS.items() if key in all_labels}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', alpha=0.5) for color in anti_colors.values()]
    plt.legend(markers, anti_colors.keys(), numpoints=1, loc="lower right")
    axs[1].set_xlabel("Snow Depth [mm]")
    axs[1].set_ylabel("Mean Force [N]")

    # frame around observation
    for frame_axis in ['top','bottom','left','right']:
        axs[0].spines[frame_axis].set_linewidth(1.25)
        axs[0].spines[frame_axis].set_color("black")

    # add a grid and remove ticks from observation
    for ax in axs:
        ax.label_outer()
        if grid: ax.grid(color="white")

    # add title
    if title is None:
        plt.suptitle("Observed and Predicted SMP Profile {}".format(smp_name))
    else:
        plt.suptitle(title)

    # adjust spacing, especially between the plots
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.002)

    # show or save plot
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()

def smp_pair_both(smp_true, smp_pred, smp_name, title=None, file_name="output/plots_data/smp_pair_both.png"):
    """ Visualizes the prediced and the observed smp profile both in one plot.
    Parameters:
        smp_true (pd.DataFrame): Only one SMP profile -the observed one-, which is already filtered out.
        smp_pred (pd.DataFrame): Only one SMP profile -the predicted one-, which is already filtered out.
        smp_name (num or str): Name of the SMP profile that is observed and predicted.
        title (str): Title of the plot.
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    if isinstance(smp_name, str):
        smp_wanted = idx_to_int(smp_name)
    else:
        smp_wanted = smp_name

    smps = [smp_true, smp_pred]

    fig, axs = plt.subplots(2, sharex=True, sharey=True)

    for ax, smp in zip(axs, smps):
        ax = sns.lineplot(smp["distance"], smp["mean_force"], ax=ax)
        last_label_num = 1
        last_distance = -1
        # going through labels and distance
        for label_num, distance in zip(smp["label"], smp["distance"]):
            if (label_num != last_label_num):
                # assign new background for each label
                background = ax.axvspan(last_distance, distance-1, color=COLORS[last_label_num], alpha=0.5)
                last_label_num = label_num
                last_distance = distance-1

            if distance == smp.iloc[len(smp)-1]["distance"]:
                ax.axvspan(last_distance, distance, color=COLORS[label_num], alpha=0.5)
        ax.set_xlim(0, len(smp)-1)
        ax.set_ylim(0)
        ax.set(ylabel=None)

    fig.text(0.01, 0.9, "Ground Truth", fontweight="bold", fontsize=8.5, transform=axs[0].transAxes)
    fig.text(0.01, 0.9, "Prediction", fontweight="bold", fontsize=8.5, transform=axs[1].transAxes)
    all_labels = list(set([*smp_true["label"].unique(), *smp_pred["label"].unique()]))
    anti_colors = {ANTI_LABELS[key] : value for key, value in COLORS.items() if key in all_labels}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', alpha=0.5) for color in anti_colors.values()]
    plt.legend(markers, anti_colors.keys(), numpoints=1, loc="lower right")
    if title is None:
        plt.suptitle("Observed and Predicted SMP Profile {}".format(smp_name))
    else:
        plt.suptitle(title)
    fig.text(0.015,0.5, "Mean Force [N]", ha="center", va="center", rotation=90)
    plt.xlabel("Snow Depth [mm]")
    plt.tight_layout()
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()

def smp_features(smp, smp_name, features, file_name="output/plots_data/smp_features.png"):
    """ Plots all wished features in the lineplot of a single SMP Profile.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        smp_name (String): Name of the wished smp profile
        features (list): features that should be plotted into the smp profile
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    smp_profile = smp[smp["smp_idx"] == idx_to_int(smp_name)]
    smp_melted = smp_profile.melt(id_vars=["distance"], value_vars=features, var_name="Feature", value_name="Value")
    g = sns.relplot(data=smp_melted, x="distance", y="Value", hue="Feature", kind="line", height=3, aspect=2/1)
    g.fig.suptitle("{} SMP Profile Normalized Distance and Different Features\n".format(smp_name))
    plt.xlabel("Distance")
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()



if __name__ == "__main__":
    main()
