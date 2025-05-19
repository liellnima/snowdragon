import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path 
from snowmicropyn import Profile

from snowdragon import OUTPUT_DIR
from snowdragon.utils.idx_funcs import int_to_idx, idx_to_int

# important setting to scale the pictures correctly
plt.rcParams.update({"figure.dpi": 250})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def all_in_one_plot(
        smp: pd.DataFrame, 
        colors: dict,
        anti_labels_long: dict,
        show_indices: bool = False, 
        sort: bool = True, 
        title: str = "SMP Profiles with Labels",
        file_name: Path = OUTPUT_DIR / "plots_data" / "all_in_one_labels.png", 
        profile_name: str = None,
        example_smp_path: Path = None,
    ):
    """ Creates a plot where all profiles are visible with their labels.
    Plot can only be saved, not shown (GUI too slow for the plot).
    Parameters:
        smp (pd.DataFrame): SMP preprocessed data
        colors (dict): Color dictionary from configs. Matches a grain number (int) to a color: < grain(int): color(str)>
        anti_labels_long (dict): Label dictionary from configs, inversed, with  the grain types fully written out. Matches the int identifier to a string describing the grain: <int: str>
        show_indices (bool): if the SMP profile indices should be displayed
        sort (bool): if the SMP profiles should be sorted according to length
        title (str): Title of the plot
        file_name (Path): where the resulting picture should be saved
        profile_name (str): Default is None, meaning no additional profile is plotted
            within the figure. If there is a string indicating a profile this one
            is plotted within the overview plot (with arrow).
        example_smp_path (Path): Path to the profile from profile_name. Needed to retrieve the right profile.
    """
    #plt.rcParams.update({"figure.dpi": 400})
    # be aware that predictions from other models must be consistent with the labels we know
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    smp_indices = list(labelled_smp["smp_idx"].unique())
    bar_width = 0.9 / len(smp_indices)
    x_ticks = np.linspace(bar_width/2, 1.0 - (bar_width/2), num=len(smp_indices))

    # make a list where each entry is a collection of all labels (ordered) from a smp profile
    # reverse order of the labels, since the last label should be printed first
    smp_idx_labels = [labelled_smp[labelled_smp["smp_idx"] == smp_index]["label"][::-1] for smp_index in smp_indices]
    # sort the smp indices list according to length
    if sort:
        lens = [len(smp_idx_profile) for smp_idx_profile in smp_idx_labels]
        sort_indices = np.argsort(lens)
        smp_indices = [smp_indices[ix] for ix in sort_indices]
        smp_idx_labels = [smp_idx_labels[ix] for ix in sort_indices]

    # maximal found distance for all profiles
    max_distance = len(max(smp_idx_labels, key = lambda x: len(x)))
    # numpy array with 0 where no label exists anymore
    smp_idx_labels_filled = np.zeros([len(smp_idx_labels), max_distance])
    for i,j in enumerate(smp_idx_labels):
        smp_idx_labels_filled[i][0:len(j)] = j

    # iterate through each mm of all profiles
    # plot a 1mm bar and assign the label corresponding colors
    for cur_mm in tqdm(reversed(range(max_distance)), total=len(range(max_distance))):
        label_colors = [colors[cur_label] if cur_label != 0 else "white" for cur_label in smp_idx_labels_filled[:, cur_mm]]
        plt.bar(x_ticks, np.repeat(1 + cur_mm, len(smp_indices)), width=bar_width, color=label_colors)

    # producing the legend for the labels
    # remove /n from antilabels
    anti_labels_stripped = {key: value.replace("\n", ' ') for key, value in anti_labels_long.items()}
    anti_labels_stripped[7] = "Decomposed and Fragmented\nPrecipitation Particles"
    anti_labels_stripped[13] = "Melted Form Clustered Rounded\nGrains"
    anti_colors = {anti_labels_stripped[key] : value for key, value in colors.items() if key in labelled_smp["label"].unique()}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in anti_colors.values()]
    plt.yticks(range(0, max_distance, 100))
    plt.legend(markers, anti_colors.keys(), numpoints=1,
               title="Snow Grain Types", loc="center left",
               bbox_to_anchor=(1, 0.5), handletextpad=0.8, labelspacing=0.8,
               frameon=False, title_fontsize=14)#, markerscale=3)
    plt.ylabel("Distance from Ground [mm]", fontsize=14)

    if title is not None:
        plt.title(title)

    # get the labels
    labels = [int_to_idx(smp_index) for smp_index in smp_indices]

    if show_indices:
        #labels = [str(int(smp_index)) for smp_index in smp_indices]
        plt.xticks(labels=labels, ticks=x_ticks, rotation=90, fontsize=3)
    else:
        plt.xticks([])

    plt.xlabel("Snow Micro Pen Profiles", fontsize=14)
    plt.xlim(0.0, 1.0)
    plt.ylim(0, int(math.ceil(max_distance / 100.0)) * 100) # rounds up to next hundred

    # add plot within plot
    ax = plt.gca()
    fig = plt.gcf()
    ax_in_plot = ax.inset_axes([0.15,0.5,0.4,0.4])
    if profile_name:
        # retrieve original smp signal
        # load npz in smp_profiles_updated
        raw_file = Profile.load(example_smp_path / (str(profile_name) + ".pnt"))
        raw_profile = raw_file.samples_within_snowpack(relativize=True)
        sns.lineplot(data=raw_profile, x="distance", y="force", ax=ax_in_plot, color="darkgrey")

    if isinstance(profile_name, str):
        smp_wanted = idx_to_int(profile_name)
    else:
        smp_wanted = profile_name

    smp_profile = smp[smp["smp_idx"] == smp_wanted]

    if profile_name:
        sns.lineplot(data=smp_profile, x="distance", y="mean_force", ax=ax_in_plot)# , color="darkslategrey"
        ax_in_plot.set_xlabel("Distance from Surface [mm]")
        ax_in_plot.set_ylabel("Mean Force [N]")
        ax_in_plot.set_xlim(0, len(smp_profile)-1)
        ax_in_plot.set_ylim(0, 10)
        ax_in_plot.set_title("Snow Micro Pen Signal") #of\nProfile {}".format(profile_name)

    # add background colors!
    last_label_num = 1
    last_distance = -1
    for label_num, distance in zip(smp_profile["label"], smp_profile["distance"]):
        if (label_num != last_label_num):
            # assign new background for each label
            background = ax_in_plot.axvspan(last_distance, distance-1, color=colors[last_label_num], alpha=0.5)
            last_label_num = label_num
            last_distance = distance-1

        if distance == smp_profile.iloc[len(smp_profile)-1]["distance"]:
            ax_in_plot.axvspan(last_distance, distance, color=colors[label_num], alpha=0.5)

    if profile_name:
        # find location of smp profile
        profile_loc = (labels.index(profile_name) / len(labels)) + (bar_width*1.5)
        # draw arrow between plot and smp profile
        ax.annotate("", xy=(profile_loc, 80), xytext=(0.55, 400), arrowprops=dict(shrink=0.05)) # facecolor="black",
    fig.set_size_inches(10, 5) # set size of figure
    #plt.savefig(file_name, bbox_inches="tight", dpi=300)
    #ax.set_aspect(aspect=0.5)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.close()


def bog_plot(
        smp: pd.DataFrame, 
        sort: bool = True, 
        file_name: Path = OUTPUT_DIR / "plots_data" / "bogplot.png"
    ):
    """ Creates a bog plot for the given smp profiles. Makes the mean force visible.
    Parameters:
        smp (pd.DataFrame): dataframe containing smp profiles
        sort (bool): indicates if profiles should be sorted ascending
        file_name (Path): If None - plot is shown. If str, the file is saved there.
    """
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    distance_between_smp = 0.5
    day_id = 1
    smp_indices_unsorted = labelled_smp["smp_idx"].unique()
    # sort the smp indices according to length
    if sort:
        smp_lengths = np.array([labelled_smp[labelled_smp["smp_idx"] == smp_idx]["distance"].max() for smp_idx in smp_indices_unsorted])
        smp_indices = [smp_indices_unsorted[ix] for ix in np.argsort(smp_lengths)]
    else:
        smp_indices = smp_indices_unsorted

    # apply logarithm to force
    #labelled_smp.loc[:, "mean_force"] = labelled_smp.loc[:, "mean_force"].apply(lambda x: np.log10(x))
    for i, curr_smp_idx in zip(range(len(smp_indices)), smp_indices):
        smp_profile = labelled_smp[labelled_smp["smp_idx"] == curr_smp_idx]
        Y = smp_profile["mean_force"][::-1]
        z = smp_profile["distance"]
        #contour_levels = np.arange( 0, 3, 0.5)
        #contour_levels[-1] = 3
        #contour_levels = np.arange( 0, 34, 0.5)
        contour_levels = np.arange( 0, 1, 0.05)
        x1 = i * distance_between_smp
        x2 = (i+1) * distance_between_smp
        plt.contourf([x1, x2], z, np.array([Y,Y]).transpose(), levels=contour_levels, cmap="jet")# np.array([Y,Y]).transpose(), cmap="jet")

    plt.xlabel("SMP Profile Indices")
    plt.ylabel("Distance from Ground")
    plt.xticks([])
    #plt.gca().invert_yaxis()
    #plt.tight_layout()
    cbar = plt.colorbar()
    cbar.set_label("Mean force", rotation=90)
    #cbar.set_ticks(np.arange(0, 1.5, 0.5))
    plt.title("All Labelled SMP Profiles with Normalized Force Values")
    #plt.grid()
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()

