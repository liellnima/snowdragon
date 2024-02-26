# import from other snowdragon modules
from data_handling.data_preprocessing import idx_to_int
from models.helper_funcs import int_to_idx
from visualization.plot_profile import smp_unlabelled
from data_handling.data_parameters import LABELS, ANTI_LABELS, COLORS, ANTI_LABELS_LONG

import os
import io
import math
import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import matplotlib.ticker as ticker


from tqdm import tqdm
from scipy import stats
from tabulate import tabulate
from snowmicropyn import Profile
from sklearn.tree import export_graphviz
from sklearn.tree._tree import TREE_LEAF
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_curve, auc

import matplotlib.lines as lines

# important setting to scale the pictures correctly
plt.rcParams.update({"figure.dpi": 250})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def plot_balancing(smp, title="Distribution of Labels in the Labelled SMP Dataset",
    file_name="output/plots_data/balance_of_dataset.svg"):
    """ Produces a plot that shows how balanced the dataset is.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        title (str): Title. No title if None.
        file_name (str): If None - plot is shown. If str, the file is saved there.
    """
    # take only labelled data and exclude surface and ground
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]
    # I can currently not find snow-ice because I am cutting of the last datapoints, if they are less than 1mm
    print("Can I find the snow-ice label?", smp[smp["label"] == 11])

    colors = [COLORS[label_number] for label_number in labelled_smp["label"].value_counts().index]
    ax = sns.countplot(x="label", data=labelled_smp, order=labelled_smp["label"].value_counts().index, palette=colors)
    if title is not None:
        plt.title(title)
    plt.xlabel("Labels")
    plt.box(False)
    ax2=ax.twinx()
    ax2.set_ylabel("Frequency [%]")
    ax2.spines['right'].set_visible(True)
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate("{:.1f}%".format(100.*y/len(labelled_smp)), (x.mean(), y), ha="center", va="bottom") # set the alignment of the text
    x_labels = [ANTI_LABELS[label_number] for label_number in labelled_smp["label"].value_counts().index]
    ax.set_xticklabels(x_labels, rotation=0)
    ax2.set_ylim(0,50)
    ax.set_ylim(0,len(labelled_smp)*0.5)

    # legend
    anti_colors = {ANTI_LABELS_LONG[key] : value for key, value in COLORS.items() if key in labelled_smp["label"].unique()}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', alpha=1) for color in anti_colors.values()]
    plt.legend(markers, anti_colors.keys(), numpoints=1, loc="upper right", fontsize=8)

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()

def all_in_one_plot(smp, show_indices=False, sort=True, title="SMP Profiles with Labels",
    file_name="output/plots_data/all_in_one_labels.png", profile_name=None):
    """ Creates a plot where all profiles are visible with their labels.
    Plot can only be saved, not shown (GUI too slow for the plot).
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        show_indices (bool): if the SMP profile indices should be displayed
        sort (bool): if the SMP profiles should be sorted according to length
        title (str): Title of the plot
        file_name (str): where the resulting picture should be saved
        profile_name (str): Default is None, meaning no additional profile is plotted
            within the figure. If there is a string indicating a profile this one
            is plotted within the overview plot (with arrow).
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
        label_colors = [COLORS[cur_label] if cur_label != 0 else "white" for cur_label in smp_idx_labels_filled[:, cur_mm]]
        plt.bar(x_ticks, np.repeat(1 + cur_mm, len(smp_indices)), width=bar_width, color=label_colors)

    # producing the legend for the labels
    # remove /n from antilabels
    anti_labels_stripped = {key: value.replace("\n", ' ') for key, value in ANTI_LABELS_LONG.items()}
    anti_labels_stripped[7] = "Decomposed and Fragmented\nPrecipitation Particles"
    anti_labels_stripped[13] = "Melted Form Clustered Rounded\nGrains"
    anti_colors = {anti_labels_stripped[key] : value for key, value in COLORS.items() if key in labelled_smp["label"].unique()}
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
    if profile_name == "S31H0368":
        # retrieve original smp signal
        # load npz in smp_profiles_updated
        raw_file = Profile.load("../Data/Arctic_updated/sn_smp_31/exdata/PS122-3_30-42/" + profile_name + ".pnt")
        raw_profile = raw_file.samples_within_snowpack(relativize=True)
        sns.lineplot(data=(raw_profile["distance"], raw_profile["force"]), ax=ax_in_plot, color="darkgrey")

    if isinstance(profile_name, str):
        smp_wanted = idx_to_int(profile_name)
    else:
        smp_wanted = profile_name

    smp_profile = smp[smp["smp_idx"] == smp_wanted]

    sns.lineplot(data=(smp_profile["distance"], smp_profile["mean_force"]), ax=ax_in_plot)# , color="darkslategrey"
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
            background = ax_in_plot.axvspan(last_distance, distance-1, color=COLORS[last_label_num], alpha=0.5)
            last_label_num = label_num
            last_distance = distance-1

        if distance == smp_profile.iloc[len(smp_profile)-1]["distance"]:
            ax_in_plot.axvspan(last_distance, distance, color=COLORS[label_num], alpha=0.5)


    # find location of smp profile
    profile_loc = (labels.index(profile_name) / len(labels)) + (bar_width*1.5)
    # draw arrow between plot and smp profile
    ax.annotate("", xy=(profile_loc, 80), xytext=(0.55, 400), arrowprops=dict(shrink=0.05)) # facecolor="black",
    fig.set_size_inches(10, 5) # set size of figure
    #plt.savefig(file_name, bbox_inches="tight", dpi=300)
    #ax.set_aspect(aspect=0.5)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.close()

# Longterm TODO: more beautiful heatmaps: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
def corr_heatmap(smp, labels=None, file_name="output/plots_data/corr_heatmap.png", title=""):
    """ Plots a correlation heatmap of all features.
    Parameters:
        smp (df.Dataframe): SMP preprocessed data
        labels (list): Default None - usual complete correlation heatmap is calculated.
            Else put in the labels for which the correlation heatmap should be calculated
        file_name (str): where the resulting pic should be saved
        title (str): title of the figure
    """
    if labels is None:
        smp_filtered = smp.drop("label", axis=1)
        smp_corr = smp_filtered.corr()
        mask = np.triu(np.ones_like(smp_corr, dtype=np.bool))
        mask = mask[1:, :-1]
        corr = smp_corr.iloc[1:, :-1].copy()
        sns.heatmap(corr, mask=mask, annot=False, fmt=".2f", vmin=-1, vmax=1,
                    center=0, annot_kws={"fontsize": 5},
                    cbar_kws={"label": "Pearson Correlation"})
        plt.xticks(range(0, 24),
                   ["dist", "mean", "var", "min", "max", "mean_4", "var_4",
                    "min_4", "max_4", "med_4", "lambda_4", "delta_4", "L_4",
                    "mean_12", "var_12", "min_12", "max_12", "med_12",
                    "lambda_12", "delta_12", "L_12", "gradient", "smp_idx", "pos_rel"],
                    fontsize=7, rotation=45)
        plt.yticks(range(0, 24),
                   ["mean", "var", "min", "max", "mean_4", "var_4",
                    "min_4", "max_4", "med_4", "lambda_4", "delta_4", "L_4",
                    "mean_12", "var_12", "min_12", "max_12", "med_12",
                    "lambda_12", "delta_12", "L_12", "gradient", "smp_idx", "pos_rel", "dist_gro"],
                    fontsize=7)
        plt.tight_layout(rect=[-0.02, 0, 1.07, 0.95])
        plt.title(title) #"Correlation Heatmap of SMP Features"
        if file_name is not None:
            plt.savefig(file_name, dpi=200)
            plt.close()
        else:
            plt.show()
    else:
        col_names = []
        # this makes only sense for labelled data
        smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]
        smp_labelled = smp_labelled.drop("smp_idx", axis=1)
        for label in labels:
            # make the label integer if it not already
            if not isinstance(label, int): label = LABELS[label]
            # add the label to smp_labelled
            col_name = ANTI_LABELS[label]
            col_names.append(col_name)
            smp_labelled[col_name] = (smp_labelled["label"] == label) * 1
        # drop label columns
        smp_labelled = smp_labelled.drop("label", axis=1)
        # calculate the correlation heatmap
        smp_corr = smp_labelled.corr() # Pearson Correlation
        # consider only the correlations between labels and features
        corr = smp_corr.iloc[-len(labels):, :].copy()
        corr = corr.drop(col_names, axis=1)
        # plot the resulting heatmap
        sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0,
                    annot_kws={"fontsize":6}, cmap="RdBu_r", #RdBu_r #coolwarm
                    cbar_kws={"label": "Pearson Correlation", "pad":0.02})
        plt.tight_layout(rect=[0.01, -0.05, 1.07, 0.95]) #[0, -0.05, 1.07, 0.95]
        plt.xticks(range(0, 24),
                   ["dist", "mean", "var", "min", "max", "mean 4", "var 4",
                    "min 4", "max 4", "med 4", "lambda 4", "delta 4", "L 4",
                    "mean 12", "var 12", "min 12", "max 12", "med 12",
                    "lambda 12", "delta 12", "L 12", "gradient", "pos rel", "dist gro"],
                   rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel("Features of Snow Micro Pen Data")
        plt.ylabel("Snow Grain Types")
        plt.title(title)#"Correlation Heat Map of SMP Features with Different Labels"
        if file_name is not None:
            plt.savefig(file_name, dpi=300)
            plt.close()
        else:
            plt.show()


def anova(smp, file_name=None, tablefmt='psql'):
    """ Prints ANOVA F-scores for features.
    Parameters:
        smp (df.Dataframe): SMP preprocessed data
        file_name (str): in case the results should be saved in a file, indicate the path here
        tablefmt (str): table format that should be used for tabulate, e.g. 'psql' or 'latex_raw'
    """
    np.set_printoptions(precision=3)
    smp_filtered = smp[smp["label"] != 0]
    features = smp_filtered.drop("label", axis=1)
    target = smp_filtered["label"]
    # feature extraction with ANOVA (f_classif)
    test = SelectKBest(score_func=f_classif, k="all")
    fit = test.fit(features, target)
    results = pd.DataFrame({"Feature" : features.columns, "ANOVA-F-value" : fit.scores_, "P-value" : fit.pvalues_})
    results = results.sort_values(by=["ANOVA-F-value"], ascending=False)
    results = results.reset_index(drop=True)

    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(tabulate(results, headers='keys', tablefmt=tablefmt))

    print("ANOVA Feature Ranking")
    print(tabulate(results, headers='keys', tablefmt=tablefmt))


def forest_extractor(smp, file_name=None, tablefmt="psql", plot=False):
    """ Random Forest for feature extraction.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        file_name (str): in case the results should be saved in a file, indicate the path here
        tablefmt (str): table format that should be used for tabulate, e.g. 'psql' or 'latex_raw'
        plot (bool): shows a plotbar with the ranked feature importances
    """
    smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    x = smp_labelled.drop(["label", "smp_idx"], axis=1)
    y = smp_labelled["label"]
    # Build a forest and compute the impurity-based feature importances
    forest = RandomForestClassifier(n_estimators=250,
                                  random_state=42)
    forest.fit(x, y)
    importances = forest.feature_importances_

    # if plot:
    #     # Plot feature importances as pixels
    #     plt.matshow(importances, cmap=plt.cm.hot)
    #     plt.title("Feature importances with forests of trees")
    #     plt.show()

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    indices_names = [list(x.columns)[index] for index in indices]

    # Print the feature ranking
    importance_list = [importances[indices[f]] for f in range(x.shape[1])]
    results = pd.DataFrame({"Feature" : indices_names, "Tree-Importance" : importance_list})

    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(tabulate(results, headers="keys", tablefmt=tablefmt))

    print("Decision Tree Feature Ranking:")
    print(tabulate(results, headers='keys', tablefmt=tablefmt))

    # Plot the impurity-based feature importances of the forest
    if plot:
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(x.shape[1]), importances[indices],
                color="tab:green", yerr=std[indices], align="center")
        indices_names = ["dist_gro", "pos_rel", "dist", "L_4", "var_12",
                         "min_4", "min_12", "L_12", "var_4", "med_12",
                         "min", "med_4", "var", "max_12", "max_4", "mean_12",
                         "mean_4", "mean", "max", "delta_12", "gradient",
                         "delta_4", "lambda_12", "lambda_4"]
        plt.xticks(range(x.shape[1]), indices_names, rotation=90, fontsize=8)
        plt.xlim([-1, x.shape[1]])
        plt.tight_layout()
        plt.show()

def pairwise_features(smp, features, samples=None, kde=False, file_name="output/plots_data/pairwise_features.png"):
    """ Produces a plot that shows the relation between all the feature given in the features list.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        features (list): contains all features that should be displayed for pairwise comparison
        samples (int): Default None, how many samples should be drawn from the labelled dataset
        kde (bool): Default False, whether the lower triangle should overlay kde plots
        file_name (str): where the pic should be saved. If 'None' the plot is shown.
    """
    # use only data that is already labelled
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]
    smp_filtered = labelled_smp[features] if samples is None else labelled_smp[features].sample(n=samples, random_state=42)
    g = sns.pairplot(smp_filtered, hue="label", palette=COLORS, plot_kws={"alpha": 0.5, "linewidth": 0})
    if kde : g.map_lower(sns.kdeplot, levels=4, color=".2")
    new_labels = [ANTI_LABELS[int(float(text.get_text()))] for text in g._legend.texts]
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    if file_name is not None:
        plt.savefig(file_name, dpi=200)
        plt.close()
    else:
        plt.show()



def bog_plot(smp, sort=True, file_name="output/plots_data/bogplot.png"):
    """ Creates a bog plot for the given smp profiles. Makes the mean force visible.
    Parameters:
        smp (pd.DataFrame): dataframe containing smp profiles
        sort (bool): indicates if profiles should be sorted ascending
        file_name (str): If None - plot is shown. If str, the file is saved there.
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

def prune(decisiontree, min_samples_leaf = 1):
    """ Function for posterior decision tree pruning.
    Paramters:
        decisiontree (sklearn.tree): The decision tree to prune
        min_samples_leaf (int): How many samples should be sorted to this leaf minimally?
    """
    if decisiontree.min_samples_leaf >= min_samples_leaf:
        raise Exception('Tree already more pruned')
    else:
        decisiontree.min_samples_leaf = min_samples_leaf
        tree = decisiontree.tree_
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            if n_samples <= min_samples_leaf:
                tree.children_left[i]=-1
                tree.children_right[i]=-1

def visualize_tree(rf, x_train, y_train, feature_names=None, tree_idx=0, min_samples_leaf=1000, file_name="output/tree", format="png"):
    """ Visualizes a single tree from a decision tree. Works only explicitly for my current data.
    Parameters:
        rf (RandomForestClassifier): the scikit learn random forest classfier
        x_train (pd.DataFrame): Input data for training. If None the rf is pretrained.
        y_train (pd.Series): Target data for training. If None the rf is pretrained.
        feature_names (list): Default None, since this is assigned from training data.
            If the rf is pretrained, this must be assigned here. (e.g. smp.columns)
        tree_idx (int): Indicates which tree from the random forest should be visualized?
        min_samples_leaf (int): Indicates how many samples should be sorted to a leaf minimally
        file_name (str): The name under which the resulting png should be saved (without extension!)
        format (str): e.g. png or svg, indicates how the pic should be stored
    """

    if (x_train is not None) and (y_train is not None):
        # deciding directly which label gets which decision tree label
        y_train[y_train==6.0] = 0
        y_train[y_train==3.0] = 1
        y_train[y_train==5.0] = 2
        y_train[y_train==12.0] = 3
        y_train[y_train==4.0] = 4
        y_train[y_train==17.0] = 5
        y_train[y_train==16.0] = 6
        anti_labels = {0:"rgwp", 1:"dh", 2: "mfdh", 3:"dhwp", 4:"dhid", 5:"rare", 6:"pp"}
        class_names = [anti_labels[label] for label in y_train.unique()]

        feature_names = x_train.columns

        # fit the model
        rf.fit(x_train, y_train)

    else:
        feature_names = feature_names
        class_names = [ANTI_LABELS[c] for c in rf.classes_]

    # extract one decision tree
    estimator = rf.estimators_[tree_idx]
    # we have to prune the tree otherwise the tree is way too big
    prune(estimator, min_samples_leaf=min_samples_leaf)

    # export image as dot file
    dot_data = export_graphviz(estimator, out_file = None, #file_name + ".dot",
                feature_names = feature_names,
                class_names = class_names,
                rounded = True, proportion = True,
                precision = 2, filled = True, rotate=False)

    new_dot_data = "\\n".join([line for line in dot_data.split("\\n") if not line.startswith("value")])
    # save that as png
    graphviz.Source(new_dot_data, format=format).render(filename = file_name)
    os.system("rm " + file_name)
    # make a png file from the dot file and delete the dot file
    # os.system("dot -Tpng "+ file_name + ".dot -o " + file_name + ".png")
    # os.system("rm " + file_name + ".dot")


def plot_confusion_matrix(confusion_matrix, labels, name="", file_name="output/plots_data/confusion_matrix.png"):
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

def plot_roc_curve(y_trues, y_prob_preds, labels, name="", file_name="output/plots_data/roc_curve.png"):
    """ Plotting ROC curves for all labels of a multiclass classification problem.
    Inspired from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    Parameters:
        y_trues
        y_prob_preds
        labels
        name
        file_name (str): path where the plot should be saved. If None the plot is
            shown and not stored.
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

    # Plot all ROC curves
    plt.figure()
    plt.plot(false_pos_rate["micro"], true_pos_rate["micro"],
             label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
             color="deeppink", linestyle=':', linewidth=4)

    plt.plot(false_pos_rate["macro"], true_pos_rate["macro"],
             label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
             color="blueviolet", linestyle=':', linewidth=4)

    colors = [COLORS[label] for label in labels]
    for i, color, label in zip(range(n_classes), colors, labels):
        plt.plot(false_pos_rate[i], true_pos_rate[i], color=color, lw=2,
                 label="ROC curve of class {0} (area = {1:0.2f})".format(ANTI_LABELS[label], roc_auc[i]))

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
