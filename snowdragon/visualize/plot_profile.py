import glob
import math
import pickle
import random
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

from tqdm import tqdm
from snowdragon.utils.helper_funcs import idx_to_int
#from data_handling.data_parameters import COLORS, ANTI_LABELS, ANTI_LABELS_LONG, LABELS, USED_LABELS, RARE_LABELS

# TODO
# make function to compare predicted profiles where we dont have the ground truth!
# adapt compare_plot accordingly --> essentially: do this with ini files only!

def plot_winter_profiles(model_name, file_name):
    """ Plots bogplot for a set of winterprofiles
    """
    np.random.seed(42)
    # read out what are winter profiles
    meta = pd.read_csv("data/metadata.csv")[["Event", "PNT_file"]]
    meta["Event"] = meta["Event"].str.split(r"_", expand=True)
    # exclude non pnt files
    pnts_mask = [True if ".pnt" in pnt else False for pnt in list(meta["PNT_file"])]
    meta = meta[pnts_mask]
    new_pnts = [pnt.split(".")[0] for pnt in list(meta["PNT_file"])] #.split("_")[0]
    meta["PNT_file"] = new_pnts

    # only include PS122_1, PS122_2
    # 4 is not winter season, 3 is our training data set and we want to go out of dist
    meta = meta[(meta["Event"] != "PS122-4") & (meta["Event"] != "PS122-3")]

    # choose a random set of these profiles
    chosen_idx = np.random.randint(len(meta), size=103)
    chosen_profiles = list(meta.iloc[chosen_idx]["PNT_file"])

    # retrieve predictions from ini files
    preds = []
    for profile in chosen_profiles:
        try:
            with open("output/predictions/{}/{}.ini".format(model_name, profile), "rb") as handle:
                markers = str(handle.read())
                new_markers = markers.split("\\n")[1:-2]
                dict_markers = {marker.split(" = ")[0]: marker.split(" = ")[1] for marker in new_markers}
                # relativize predictions
                sfc_value = dict_markers["surface"]
                ground_value = float(dict_markers["ground"]) - float(sfc_value)
                rel_pred = {label: float(value) - float(sfc_value) for label, value in dict_markers.items()}
                rel_pred = {label: ground_value - value for label, value in rel_pred.items()}
                preds.append(rel_pred)
        except OSError as e:
            print("Skipped file that couldn't be found in the predictions.")

    # translate those in some plot
    ini_bogplots(preds, chosen_profiles, "LSTM", file_name)

def ini_bogplots(preds, chosen_profiles, model_name, file_name):
    """
    """
    fig, ax = plt.subplots()

    smps = preds

    # length of bars
    bar_width = 0.9 / len(smps)
    x_ticks = np.linspace(bar_width/2, 1.0 - (bar_width/2), num=len(smps))

    # sort the smp indices list according to length
    lens = [max(smp.values()) for smp in smps]
    sort_indices = np.argsort(lens)
    smps_sorted = [smps[ix] for ix in sort_indices] # only labels please!

    # maximal found distance for all profiles
    max_distance = len(max(smps_sorted, key = lambda x: len(x)))

    # TODO CONTINUE HERE (is a dictionary the right things here? not ordered!)
    # iterate through each profile
    for i, profile in enumerate(smps_sorted):
        profile_sorted = dict(sorted(profile.items(), key=lambda item: item[1], reverse=True))
        profile_sorted = [(key, value) for key, value in profile_sorted.items()]
        for j, (label, height) in enumerate(profile_sorted[0:-1]):
            this_label = profile_sorted[j+1][0]
            label_color = COLORS[LABELS[this_label]]
            ax.bar(i/100, height, width=bar_width, color=label_color)

    # plot bars over each other for each profile for the different labels
    # sort dictionary entries according to size
    # start with highest value and plot
    # continue until lowest value

    # add label about model
    ax.text(0.5, 0.95, model_name, fontsize=14, verticalalignment="top", ha="center", transform=ax.transAxes)

    # producing the legend for the labels
    # remove /n from antilabels
    # Snow Grain Legend
    #USED_LABELS = [6.0, 3.0, 4.0, 12.0, 5.0, 16.0] # no rare
    anti_colors = {ANTI_LABELS_LONG[key] : value for key, value in COLORS.items() if key in USED_LABELS}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in anti_colors.values()]

    #plt.yticks(range(0, max_distance, 100))
    fig.legend(markers, anti_colors.keys(), numpoints=1,
               title="Snow Grain Types", loc="upper center", bbox_to_anchor=(0.5, 0.06), handletextpad=0.8, labelspacing=0.8,
               frameon=False, title_fontsize=14, ncol=3) # (0.5, 0.09)
    plt.ylabel("Distance from Ground [mm]", fontsize=14)
    plt.xlabel("Snow Micro Pen Profiles\n", fontsize=14)

    plt.xticks([])
    plt.xlim(-0.006, 0.996)

    fig.set_size_inches(10, 5)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.close()


def plot_bogplot_comparison(model_names, file_name):
    """ Plots bogplot for different model predictions of complete test data set.

    Parameter:
        model_names (list<str>): From which models we would like to get the predictions
        file_name (str): under which name the plot should be stored
    """
    all_smps = []
    first_true_labels = True # flag: append true profile only on time

    for model in model_names:
        # list all profiles of that model
        files = glob.glob("visualization/storage/*_{}_*.pickle".format(model))
        pred_profiles = []
        true_profiles = []
        # iterate through all smp profiles of that model
        for profile in files:
            with open(profile, "rb") as handle:
                true, pred = pickle.load(handle)
                pred_profiles.append(pred)
                if first_true_labels:
                    true_profiles.append(true)

        if first_true_labels:
            all_smps.append(true_profiles)
        all_smps.append(pred_profiles)
        first_true_labels = False

    compare_bogplots(all_smps, model_names, file_name)

def plot_model_profile_comparison(model_names, file_name):
    """ Plots model predictions for three random profiles of different depths.

    Parameter:
        model_names (list<str>): From which models we would like to get the predictions
        file_name (str): under which name the plot should be stored
    """
    ## Choose your profiles ##
    profiles = choose_profiles()

    ## Choose your models ##
    # read in example data and extract prediction and true data

    ## Get Data ##
    all_smp_preds = []
    all_smp_trues = []

    for profile in profiles:
        profile_preds = []
        profile_true = []
        first_true_labels = True # flag: append true profile only on time

        for model in model_names:
            with open("visualization/storage/smp_{}_{}.pickle".format(model, profile), "rb") as handle:
                true, pred = pickle.load(handle)
                profile_preds.append(pred)
                if first_true_labels:
                    profile_true.append(true)
                    first_true_labels = False
        all_smp_preds.append(profile_preds)
        all_smp_trues.append(profile_true)

    compare_model_and_profiles(all_smp_trues, all_smp_preds, profiles, model_names, title="", grid=True, file_name=file_name)

def choose_profiles():
    """ Choose three random profiles according to length (shallow, medium, deep)
    """
    # how can we get the idx of the profile and information about its length?
    random.seed(42)
    lengths = []
    names = []
    # go through all "Majority Vote" pickles and extract length of each profile
    for file in glob.glob("visualization/storage/*_Majority Vote_*.pickle"):
        with open(file, "rb") as handle:
            length = len(pickle.load(handle)[0])
            lengths.append(length)
        name = file.split('_')[-1].split('.')[0]
        names.append(name)
        #length_dic[name] = length

    # choose a random value between 0 - 200
    # get indices that fulfill this
    idx200 = [i for i, len in enumerate(lengths) if len <= 200]
    profile200 = names[random.choice(idx200)]

    # choose a random value between 200 - 500
    idx500 = [i for i, len in enumerate(lengths) if (len > 200) and (len <= 500)]
    profile500 = names[random.choice(idx500)]

    # choose a random value between 500 - 800
    idx800 = [i for i, len in enumerate(lengths) if len > 500]
    profile800 = names[random.choice(idx800)]

    return [profile200, profile500, profile800]

def compare_bogplots(all_smps, model_names, file_name="output/results/compare_bogplots.png"):
    """ Creates bogplots for different predictions
    Parameters:
        all_smps (list<df.DataFrame>): SMP preprocessed data - each entry for one model
        file_name (str): where the resulting picture should be saved
    """
    fig, axs = plt.subplots(2, int((len(model_names)+1)/2),
                            sharex=True, sharey=True, dpi=300)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    model_names = ["Ground Truth"] + model_names

    ground_truth_colors = []

    for i_model, ax in enumerate(axs.reshape(-1)):
        model_name = model_names[i_model]
        smps = all_smps[i_model]

        # length of bars
        bar_width = 0.9 / len(smps)
        x_ticks = np.linspace(bar_width/2, 1.0 - (bar_width/2), num=len(smps))

        # sort the smp indices list according to length
        lens = [len(smp) for smp in smps]
        sort_indices = np.argsort(lens)
        smps_sorted = [smps[ix]["label"][::-1] for ix in sort_indices] # only labels please!

        # TO DELETE
        # make a list where each entry is a collection of all labels (ordered) from a smp profile
        # reverse order of the labels, since the last label should be printed first
        # smp_idx_labels = [labelled_smp[labelled_smp["smp_idx"] == smp_index]["label"][::-1] for smp_index in smp_indices]

        # maximal found distance for all profiles
        max_distance = len(max(smps_sorted, key = lambda x: len(x)))
        # numpy array with 0 where no label exists anymore (whitespace above the bars)
        smp_idx_labels_filled = np.zeros([len(smps_sorted), max_distance])
        for i,j in enumerate(smps_sorted):
            smp_idx_labels_filled[i][0:len(j)] = j

        # iterate through each mm of all profiles
        # plot a 1mm bar and assign the label corresponding colors
        mm_counter = 0
        for cur_mm in tqdm(reversed(range(max_distance)), total=len(range(max_distance))):
            label_colors = [COLORS[cur_label] if cur_label != 0 else "white" for cur_label in smp_idx_labels_filled[:, cur_mm]]
            #tops = np.repeat(1 + cur_mm, len(smps))
            bottoms = np.repeat(cur_mm, len(smps))
            heights = np.repeat(1, len(smps))

            # save ground truth label colors here during the first round!
            if i_model == 0:
                ground_truth_colors.append(label_colors.copy())

            # compare labels between ground truth and current labels
            # TODO CONTINUE HERE - comparison does not work!!!
            correct_labels = [(true_color == pred_color) for true_color, pred_color in zip(ground_truth_colors[mm_counter], label_colors)]
            wrong_labels = [not label for label in correct_labels]
            # true predictions - full alpha
            ax.bar(np.array(x_ticks)[correct_labels],
                   height=np.array(heights)[correct_labels],
                   bottom=np.array(bottoms)[correct_labels],
                   color=np.array(label_colors)[correct_labels],
                   width=bar_width, alpha=1)
            ax.bar(np.array(x_ticks)[wrong_labels],
                   height=np.array(heights)[wrong_labels],
                   bottom=np.array(bottoms)[wrong_labels],
                   color=np.array(label_colors)[wrong_labels],
                   width=bar_width, alpha=0.25)
            mm_counter += 1
        # add label about model
        ax.text(0.5, 0.95, model_name, fontsize=14, verticalalignment="top", ha="center", transform=ax.transAxes)

    # producing the legend for the labels
    # remove /n from antilabels
    # Snow Grain Legend
    # labels used: [6.0, 3.0, 4.0, 12.0, 5.0, 16.0, 17.0]
    anti_colors = {ANTI_LABELS_LONG[key] : value for key, value in COLORS.items() if key in USED_LABELS + RARE_LABELS}
    # two markers: correct and wrongly predicted
    markers = [(plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle=''), plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='', alpha=0.25)) for color in anti_colors.values()]
    plt.yticks(range(0, max_distance, 100))
    fig.legend(markers, anti_colors.keys(), numpoints=1,
               title="Snow Grain Types", loc="upper center",
               bbox_to_anchor=(0.5, 0.09), handletextpad=0.8, labelspacing=0.8,
               frameon=False, title_fontsize=14, ncol=4,
               handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)})#, markerscale=3)
    fig.text(0.06, 0.48, "Distance from Ground [mm]",  fontsize=14, va="center", rotation="vertical")
    fig.text(0.5, 0.095, "Snow Micro Pen Profiles", fontsize=14, ha="center", va="center")

    plt.xticks([])
    plt.xlim(0.0, 1.0)
    plt.ylim(0, int(math.ceil(max_distance / 100.0)) * 100) # rounds up to next hundred

    #fig.subplots_adjust(bottom=0.15)

    fig.set_size_inches(10, 10) # set size of figure
    #plt.savefig(file_name, bbox_inches="tight", dpi=300)
    #ax.set_aspect(aspect=0.5)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.close()

def compare_model_and_profiles(smp_trues, smp_preds, smp_names, model_names, title=None, grid=True, file_name="output/plots_data/smp_compare.png"):
    """
    """
    # TODO GET RED LINE FROM THIRD PLOT INTO THE CENTER
    fig, axs = plt.subplots(len(model_names)+1, len(smp_names),
                            sharex="col", sharey="col", dpi=300,
                            gridspec_kw={"width_ratios": [1.1, 2.4, 6]})
    fig.subplots_adjust(hspace=0.1, wspace=0.01) # 0.1 and 0.05
    title_names = ["Shallow", "Medium", "Deep"]
    for smp_i, smp_name in enumerate(smp_names):

        if isinstance(smp_name, str):
            smp_wanted = idx_to_int(smp_name)
        else:
            smp_wanted = smp_name

        # concatenate all the smps of this profile together
        smp_true = smp_trues[smp_i][0]
        smps = [smp_true]
        for model_preds in smp_preds[smp_i]:
            smps.append(model_preds)

        #differences = smp_true["label"] != smp_pred["label"]
        height = max(smp_true["mean_force"]) / 2

        #fig, axs = plt.subplots(3, sharex=True, sharey=True, dpi=300)
        first_ax = True
        line_handles = []
        for model_i, smp in enumerate(smps):
            # calculate differences between true and pred
            differences = smp_true["label"] != smp["label"]

            if first_ax:
                alpha = 0.4
                # plot the ground truth measurement
                #["Force Signal", "Wrong Classification"]
                signalplot = sns.lineplot(data=smp_true, x="distance", y="mean_force", ax=axs[model_i, smp_i])
                signalplot.set(ylim=height, title=title_names[smp_i])
                signalplot.set_title(title_names[smp_i], fontsize=10)
                line_handles.append(signalplot)
                first_ax = False
            else:
                alpha = 0.4
                # plot the differences
                diff = differences.astype(int) * (height) # plot line in the middle
                diff = diff.replace({0:np.inf}) # cheating a bit: seaborn wont plot inf values
                # - 0.5 is only because the plot lines have a small off-set (not numerically though)
                lineplot = sns.lineplot(data=(smp_true["distance"]-0.5, diff), ax=axs[model_i, smp_i], linewidth=2, color='r')
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
                points_heights = [height] * len(points_indices)
                scatterplot = axs[model_i, smp_i].scatter(smp["distance"][points_indices] - 0.5, points_heights, marker='s', s=3.75, edgecolors='none', color='r', alpha=1, zorder=100)

            last_label_num = 1
            last_distance = -1
            # going through labels and distance
            for label_num, distance in zip(smp["label"], smp["distance"]):
                if (label_num != last_label_num):
                    # assign new background for each label
                    background = axs[model_i, smp_i].axvspan(last_distance, distance-1, color=COLORS[last_label_num], alpha=alpha)
                    last_label_num = label_num
                    last_distance = distance-1

                if distance == smp.iloc[len(smp)-1]["distance"]:
                    axs[model_i, smp_i].axvspan(last_distance, distance, color=COLORS[label_num], alpha=alpha)

            axs[model_i, smp_i].set_xlim(0, len(smp)-1)
            axs[model_i, smp_i].set_ylim(0)
            axs[model_i, smp_i].set(ylabel=None, xlabel=None, yticklabels=[])
            axs[model_i, smp_i].tick_params(axis="y", left=False)
            #axs[model_i, smp_i].tick_params(axis="y",direction="in", pad=-22)
            #axs[model_i, smp_i].yaxis.get_major_ticks()[0].label1.set_visible(False)
            if model_i < len(model_names):
                axs[model_i, smp_i].tick_params(axis="x", bottom=False)

    # Snow Grain Legend
    # grain types
    # used labels: [6.0, 3.0, 4.0, 12.0, 5.0, 16.0, 17.0]
    anti_colors = {ANTI_LABELS_LONG[key] : value for key, value in COLORS.items() if key in USED_LABELS + RARE_LABELS}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', alpha=0.5) for color in anti_colors.values()]
    # line markers
    markers.append(mlines.Line2D([], [], color="C0", label="Force Signal"))
    markers.append(mlines.Line2D([], [], color="red", linewidth=4, label="Misclassified"))
    anti_colors["Force Signal"] = "C0"
    anti_colors["Misclassified"] = "red"
    colors_keys = list(anti_colors.keys())

    # fill in dummys
    for dummy_i in [2, 5, 8]:
        markers.insert(dummy_i, mlines.Line2D([], [], color="none", label="none"))
        colors_keys.insert(dummy_i, '')

    # make legend
    legend1 = plt.figlegend(markers, colors_keys, numpoints=1, loc=8, title="Snow Grain Types", frameon=False, ncol=4, prop={"size": 8})# borderaxespad=0
    fig.add_artist(legend1)

    if title is None:
        plt.suptitle("Observed and Predicted SMP Profile {}".format(smp_name))
    else:
        plt.suptitle(title)

    # add titles of models at the left, 90 rotated
    names = (["Ground Truth"] + model_names)[::-1]
    names = [name.replace(' ', "\n") for name in names]

    for i, model_name in enumerate(names):
        pos = i / (len(names) * 1.65) # 1.7 # 1.625
        fig.text(0.005, 0.325 + pos, model_name, rotation=90,fontsize=10) # 0.075, 0.31 # 0.95, 0.31

    # x and y labels
    #fig.text(0.02, 0.6, "Mean Force [N]", ha="center", va="center", rotation=90, fontsize=12) #0.04, 0.6
    fig.text(0.5, 0.25, "Snow Depth [mm]", ha="center", va="center", fontsize=12) #0.5 0.225

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.32, right=0.99, left=0.06) # no right and left # right=0.94, left=0.05
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()
    pass

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
            signalplot = sns.lineplot(data=smp, x="distance", y="mean_force", ax=ax)
            line_handles.append(signalplot)
            first_ax = False
        else:
            alpha = 0.2
            # plot the differences
            diff = differences.astype(int) * (max(smp["mean_force"]) / 2) # plot line in the middle
            diff = diff.replace({0:np.inf}) # cheating a bit: seaborn wont plot inf values
            # - 0.5 is only because the plot lines have a small off-set (not numerically though)
            lineplot = sns.lineplot(data=(smp["distance"]-0.5, diff), ax=ax, linewidth=4, color='r')
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
    axs[1] = sns.lineplot(data=smp_pred, x="distance", y="mean_force", ax=axs[1])
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
        ax = sns.lineplot(data=smp, x="distance", y="mean_force", ax=ax)
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

# TODO move this to helper function
def plot_single_color(hex_color, alpha):
    foreground_tuple = mcolors.hex2color(hex_color)
    foreground_arr = np.array(foreground_tuple)
    final = tuple( (1. -  alpha) + foreground_arr*alpha )
    plt.imshow([[final]])
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.show()

# Only for testing purposes!
def main():
    """ This function exists only for testing purposes.
    """
    profile_comparison = False
    bogplots = True
    winter_profiles = False

    # TODO this is done manually right now, import error for evaluate_all_models (init function must be recompiled)
    # before: run run_models.py with evaluate_all_models: one_plot = True and overwrite tables = False
    # TODO one_plot = True has to be set manually right now - change that!

    # if first_time: # needs to be only done for the first time!
    #     with open("data/preprocessed_data_k5.txt", "rb") as myFile:
    #         data = pickle.load(myFile)
    #     evaluate_all_models(data, overwrite_tables=False)
    if profile_comparison:
        plot_model_profile_comparison(model_names = ["LSTM", "Random Forest", "Self Trainer"], file_name = "output/plots_results/comparison_3_profiles_textleft.pdf")

    if bogplots:
        plot_bogplot_comparison(model_names = ["LSTM", "Random Forest", "Self Trainer"], file_name = "output/plots_results/bogplot_comparison_alpha.pdf")

    if winter_profiles:
        plot_winter_profiles(model_name="lstm", file_name = "output/plots_results/lstm_winter.pdf")
    ### FOR A SINGLE PROFILE
    # with open("visualization/storage/smp_Random Forest_S31H0234.pickle", "rb") as handle:
    #     rf_true, rf_pred = pickle.load(handle)
    # with open("visualization/storage/smp_Baseline_S31H0234.pickle", "rb") as handle:
    #     base_true, base_pred = pickle.load(handle)
    #
    # smp_preds = [base_pred, rf_pred]
    # smp_true = base_true
    #
    # ## Choose your profiles ##
    # # smp_name = "S31H0488"
    # # smp_name = "S31H0234"

    ## Compare prediction profiles for these models ##
    #compare_plot(smp_true, smp_preds, smp_name, model_names, title="", grid=True, file_name="output/plots_results/comparison_test01.png")


if __name__ == "__main__":
    main()
