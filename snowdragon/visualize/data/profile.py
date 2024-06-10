import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from snowdragon import OUTPUT_DIR
from snowdragon.utils.helper_funcs import idx_to_int

def smp_unlabelled(
        smp: pd.DataFrame, 
        smp_name: str, 
        file_name: Path = OUTPUT_DIR / "plots_data" / "smp_unlabelled.png",
    ):
    """ Plots a SMP profile without labels.
    Parameters:
        smp (pd.DataFrame): SMP preprocessed data
        smp_name (String or float/int): Name of the wished smp profile or
            alternatively its converted index number
        file_name (Path): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    if isinstance(smp_name, str):
        smp_wanted = idx_to_int(smp_name)
    else:
        smp_wanted = smp_name

    smp_profile = smp[smp["smp_idx"] == smp_wanted]
    # adapt for new seaborn version
    ax = sns.lineplot(data=smp_profile, x="distance", y="mean_force")
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


def smp_labelled(
        smp: pd.DataFrame, 
        smp_name: str, 
        colors: dict,
        anti_labels: dict,
        title: str = None, 
        file_name: Path = OUTPUT_DIR / "plots_data" / "smp_labelled.pngs",
    ):
    """ Plots a SMP profile with labels.
    Parameters:
        smp (pd.DataFrame): SMP preprocessed data
        smp_name (String or float/int): Name of the wished smp profile or
            alternatively its converted index number
        colors (dict): Color dictionary from configs. Matches a grain number (int) to a color: < grain(int): color(str)>
        anti_labels (dict): Label dictionary from configs, inversed. Matches the number identifier (int) to the string describing the grain: <int: str> 
        title (str): if None, a simple headline for the plot is used.
            Please specify with string.
        file_name (Path): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    # SHOW THE SAME PROFILE WITH LABELS
    if isinstance(smp_name, str):
        smp_wanted = idx_to_int(smp_name)
    else:
        smp_wanted = smp_name

    smp_profile = smp[smp["smp_idx"] == smp_wanted]
    ax = sns.lineplot(data=smp_profile, x="distance", y="mean_force")
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
            background = ax.axvspan(last_distance, distance-1, color=colors[last_label_num], alpha=0.5)
            last_label_num = label_num
            last_distance = distance-1

        if distance == smp_profile.iloc[len(smp_profile)-1]["distance"]:
            ax.axvspan(last_distance, distance, color=colors[label_num], alpha=0.5)


    anti_colors = {anti_labels[key] : value for key, value in colors.items() if key in smp_profile["label"].unique()}
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

def smp_features(
        smp: pd.DataFrame, 
        smp_name: str, 
        features: list, 
        file_name: Path = OUTPUT_DIR / "plots_data" / "smp_features.png",
    ):
    """ Plots all wished features in the lineplot of a single SMP Profile.
    Parameters:
        smp (pd.DataFrame): SMP preprocessed data
        smp_name (str): Name of the wished smp profile
        features (list): features that should be plotted into the smp profile
        file_name (Path): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    smp_profile = smp[smp["smp_idx"] == idx_to_int(smp_name)]
    smp_melted = smp_profile.melt(id_vars=["distance"], value_vars=features, var_name="Feature", value_name="Value")
    g = sns.relplot(data=smp_melted, x="distance", y="Value", hue="Feature", kind="line", height=3, aspect=2/1)
    g.figure.suptitle("{} SMP Profile Normalized Distance and Different Features\n".format(smp_name))
    plt.xlabel("Distance")
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()