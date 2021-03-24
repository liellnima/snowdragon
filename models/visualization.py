# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_preprocessing import idx_to_int
from data_handling.data_parameters import LABELS, ANTI_LABELS, COLORS

import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
# important setting to scale the pictures correctly
plt.rcParams.update({"figure.dpi": 250})

from tqdm import tqdm
from scipy import stats
from tabulate import tabulate
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.tree import export_graphviz
from sklearn.tree._tree import TREE_LEAF
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_curve, auc


def smp_unlabelled(smp, smp_name, save_file=None):
    """ Plots a SMP profile without labels.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        smp_name (String or float/int): Name of the wished smp profile or
            alternatively its converted index number
        save_file (str): path where the plot should be saved. If None the plot is
            shown and not stored.
    """
    if isinstance(smp_name, str):
        smp_wanted = idx_to_int(smp_name)
    else:
        smp_wanted = smp_name
    smp_profile = smp[smp["smp_idx"] == smp_wanted]
    ax = sns.lineplot(smp_profile["distance"], smp_profile["mean_force"])
    plt.title("{} SMP Profile Distance (1mm layers) and Force".format(smp_name))
    ax.set_xlabel("Snow Depth [mm]")
    ax.set_ylabel("Mean Force [N]")
    ax.set_xlim(0, len(smp_profile)-1)
    ax.set_ylim(0)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()


def smp_labelled(smp, smp_name, title=None, save_file=None):
    """ Plots a SMP profile with labels.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        smp_name (String or float/int): Name of the wished smp profile or
            alternatively its converted index number
        title (str): if None, a simple headline for the plot is used.
            Please specify with string.
        save_file (str): path where the plot should be saved. If None the plot is
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
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()



def smp_pair(smp_true, smp_pred, smp_name, title=None, grid=True, save_file=None):
    """ Visualizes the prediced and the observed smp profile in one plot.
    The observation is a bar above/inside the predicted smp profile.
    Parameters:
        smp_true (pd.DataFrame): Only one SMP profile -the observed one-, which is already filtered out.
        smp_pred (pd.DataFrame): Only one SMP profile -the predicted one-, which is already filtered out.
        smp_name (num or str): Name of the SMP profile that is observed and predicted.
        title (str): Title of the plot.
        grid (bool): If a grid should be plotted over the plot.
        save_file (str): path where the plot should be saved. If None the plot is
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
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()


def smp_pair_both(smp_true, smp_pred, smp_name, title=None, save_file=None):
    """ Visualizes the prediced and the observed smp profile both in one plot.
    Parameters:
        smp_true (pd.DataFrame): Only one SMP profile -the observed one-, which is already filtered out.
        smp_pred (pd.DataFrame): Only one SMP profile -the predicted one-, which is already filtered out.
        smp_name (num or str): Name of the SMP profile that is observed and predicted.
        title (str): Title of the plot.
        save_file (str): path where the plot should be saved. If None the plot is
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
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()


def smp_features(smp, smp_name, features):
    """ Plots all wished features in the lineplot of a single SMP Profile.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        smp_name (String): Name of the wished smp profile
        features (list): features that should be plotted into the smp profile
    """
    smp_profile = smp[smp["smp_idx"] == idx_to_int(smp_name)]
    smp_melted = smp_profile.melt(id_vars=["distance"], value_vars=features, var_name="Feature", value_name="Value")
    ax = sns.relplot(data=smp_melted, x="distance", y="Value", hue="Feature", kind="line")
    plt.title("{} SMP Profile Distance (1mm layers) and Different Features".format(smp_name))
    plt.show()

# Longterm TODO: more beautiful heatmaps: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
def corr_heatmap(smp, labels=None):
    """ Plots a correlation heatmap of all features.
    Parameters:
        smp (df.Dataframe): SMP preprocessed data
        labels (list): Default None - usual complete correlation heatmap is calculated.
            Else put in the labels for which the correlation heatmap should be calculated
    """
    if labels is None:
        smp_filtered = smp.drop("label", axis=1)
        smp_corr = smp_filtered.corr()
        mask = np.triu(np.ones_like(smp_corr, dtype=np.bool))
        mask = mask[1:, :-1]
        corr = smp_corr.iloc[1:, :-1].copy()
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f")
        plt.title("Correlation Heatmap of SMP Features")
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
        smp_corr = smp_labelled.corr()
        # consider only the correlations between labels and features
        corr = smp_corr.iloc[-len(labels):, :].copy()
        corr = corr.drop(col_names, axis=1)
        # plot the resulting heatmap
        sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0)
        plt.xticks(rotation=45)
        plt.xlabel("Features of SMP Data")
        plt.ylabel("Snow Grain Types")
        plt.title("Correlation Heat Map of SMP Features with Different Labels")
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

    if plot:
        # Plot feature importances as pixels
        plt.matshow(importances, cmap=plt.cm.hot)
        plt.title("Feature importances with forests of trees")
        plt.show()

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    indices_names = [list(x.columns)[index] for index in indices]

    # Print the feature ranking
    importance_list = [importances[indices[f]] for f in range(x.shape[1])]
    results = pd.DataFrame({"Feature" : indices_names, "Tree-Importance" : importance_list})

    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(tabulate(results, headers='keys', tablefmt=tablefmt))

    print("Decision Tree Feature Ranking:")
    print(tabulate(results, headers='keys', tablefmt=tablefmt))

    # Plot the impurity-based feature importances of the forest
    if plot:
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(x.shape[1]), importances[indices],
                color="lightgreen", yerr=std[indices], align="center")
        plt.xticks(range(x.shape[1]), indices_names, rotation=55)
        plt.xlim([-1, x.shape[1]])
        plt.show()

def pairwise_features(smp, features, samples=None, kde=False):
    """ Produces a plot that shows the relation between all the feature given in the features list.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        features (list): contains all features that should be displayed for pairwise comparison
        samples (int): Default None, how many samples should be drawn from the labelled dataset
        kde (bool): Default False, whether the lower triangle should overlay kde plots
    """
    # use only data that is already labelled
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]
    smp_filtered = labelled_smp[features] if samples is None else labelled_smp[features].sample(n=samples, random_state=42)
    g = sns.pairplot(smp_filtered, hue="label", palette=COLORS, plot_kws={"alpha": 0.5, "linewidth": 0})
    if kde : g.map_lower(sns.kdeplot, levels=4, color=".2")
    new_labels = [ANTI_LABELS[int(float(text.get_text()))] for text in g._legend.texts]
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    plt.show()

def plot_balancing(smp):
    """ Produces a plot that shows how balanced the dataset is.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    # take only labelled data and exclude surface and ground
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]
    # I can currently not find snow-ice because I am cutting of the last datapoints, if they are less than 1mm
    print("Can I find the snow-ice label?", smp[smp["label"] == 12])

    ax = sns.countplot(x="label", data=labelled_smp, order=labelled_smp["label"].value_counts().index)
    plt.title("Distribution of Labels in the Labelled SMP Dataset")
    plt.xlabel("Labels")
    ax2=ax.twinx()
    ax2.set_ylabel("Frequency [%]")
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate("{:.1f}%".format(100.*y/len(labelled_smp)), (x.mean(), y), ha="center", va="bottom") # set the alignment of the text
    x_labels = [ANTI_LABELS[label_number] for label_number in labelled_smp["label"].value_counts().index]
    ax.set_xticklabels(x_labels, rotation=90)
    ax2.set_ylim(0,100)
    ax.set_ylim(0,len(labelled_smp))
    plt.show()

def bog_plot(smp):
    """ Creates a bog plot for the given smp profiles. Makes the mean force visible.
    Parameters:
        smp (pd.DataFrame): dataframe containing smp profiles
    """
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    distance_between_smp = 0.5
    day_id = 1
    smp_indices = labelled_smp["smp_idx"].unique()
    # apply logarithm to force
    #labelled_smp.loc[:, "mean_force"] = labelled_smp.loc[:, "mean_force"].apply(lambda x: np.log10(x))
    for i, curr_smp_idx in zip(range(len(smp_indices)), smp_indices):
        smp_profile = labelled_smp[labelled_smp["smp_idx"] == curr_smp_idx]
        Y = smp_profile["mean_force"]
        z = smp_profile["distance"]
        #contour_levels = np.arange( 0, 3, 0.5)
        #contour_levels[-1] = 3
        #contour_levels = np.arange( 0, 2, 0.025)
        x1 = i * distance_between_smp
        x2 = (i+1) * distance_between_smp
        plt.contourf([x1, x2], z, np.array([Y,Y]).transpose(), cmap='jet')

    plt.xlabel('Snow Micro Pen Profiles')
    plt.ylabel('Depth (mm)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    cbar = plt.colorbar()
    cbar.set_label("Mean force (N)", rotation=90)
    plt.title("All Labelled SMP Profiles with Mean Force Values")
    plt.grid()
    plt.show()

def pca(smp, n=3, dim="both", biplot=True):
    """ Visualizing 2d and 2d plot with the 2 or 3 principal components that explain the most variance.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        n (int): how many principal components should be extracted
        dim (str): 2d, 3d or both - for visualization
        biplot (bool): indicating if the features most used for the principal components should be plotted as biplot
    """
    smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    x = smp_labelled.drop(["label", "smp_idx"], axis=1)
    y = smp_labelled["label"]
    anti_colors = {ANTI_LABELS[key] : value for key, value in COLORS.items() if key in smp_labelled["label"].unique()}
    # first two components explain the most anyway!
    pca = PCA(n_components=n, random_state=42)
    pca_result = pca.fit_transform(x)
    smp_with_pca = pd.DataFrame({"pca-one": pca_result[:,0], "pca-two": pca_result[:,1], "pca-three": pca_result[:,2], "label": y})
    print("Explained variance per principal component: {}.".format(pca.explained_variance_ratio_))
    print("Cumulative explained variance: {}".format(sum(pca.explained_variance_ratio_)))
    # 2d plot
    if dim == "2d" or dim == "both":
        g = sns.scatterplot(x="pca-one", y="pca-two", hue="label", palette=COLORS, data=smp_with_pca, alpha=0.3)
        # plot the variables that explain the highest variance
        if biplot:
            coeff = pca.components_
            labels = list(x.columns)
            for i in range(coeff.shape[0]):
                plt.arrow(0, 0, coeff[i,0], coeff[i,1], color="black", alpha=0.5, head_width=0.02)
                if labels is None:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color="black", ha='center', va='center')
                else:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color="black", ha='center', va='center')

        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', alpha=0.5) for color in anti_colors.values()]
        plt.legend(markers, anti_colors.keys(), numpoints=1, loc="center left", bbox_to_anchor=(1.04, 0.5))
        plt.show()

    # 3d plot
    if dim == "3d" or dim == "both":
        ax = plt.figure(figsize=(16,10)).gca(projection="3d")
        color_labels = [COLORS[label] for label in smp_with_pca["label"]]
        g = ax.scatter(xs=smp_with_pca["pca-one"], ys=smp_with_pca["pca-two"], zs=smp_with_pca["pca-three"], c=color_labels, alpha=0.3)
        ax.set_xlabel("pca-one")
        ax.set_ylabel("pca-two")
        ax.set_zlabel("pca-three")
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in anti_colors.values()]
        plt.legend(markers, anti_colors.keys(), numpoints=1, bbox_to_anchor=(1.04, 0.5), loc=2)
        plt.show()

def tsne(smp, dim="both"):
    """ Visualizing 2d and 2d plot with the 2 or 3 TSNE components.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        dim (str): 2d, 3d or both - for visualization
    """
    smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    x = smp_labelled.drop(["label", "smp_idx"], axis=1)
    y = smp_labelled["label"]
    anti_colors = {ANTI_LABELS[key] : value for key, value in COLORS.items() if key in smp_labelled["label"].unique()}

    if dim == "2d" or dim == "both":
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
        tsne_results = tsne.fit_transform(x)
        smp_with_tsne = pd.DataFrame({"tsne-one": tsne_results[:, 0], "tsne-two": tsne_results[:, 1], "label": y})

        sns.scatterplot(x="tsne-one", y="tsne-two", hue="label", palette=COLORS, data=smp_with_tsne, alpha=0.3)
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in anti_colors.values()]
        plt.legend(markers, anti_colors.keys(), numpoints=1, loc="center left", bbox_to_anchor=(1.04, 0.5))
        plt.show()

    if dim == "3d" or dim == "both":
        tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300, random_state=42)
        tsne_results = tsne.fit_transform(x)
        smp_with_tsne = pd.DataFrame({"tsne-one": tsne_results[:, 0], "tsne-two": tsne_results[:, 1], "tsne-three": tsne_results[:, 2], "label": y})

        ax = plt.figure(figsize=(16,10)).gca(projection="3d")
        color_labels = [COLORS[label] for label in smp_with_tsne["label"]]
        ax.scatter(xs=smp_with_tsne["tsne-one"], ys=smp_with_tsne["tsne-two"], zs=smp_with_tsne["tsne-three"], c=color_labels, alpha=0.3)
        ax.set_xlabel("tsne-one")
        ax.set_ylabel("tsne-two")
        ax.set_zlabel("tsne-three")
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', alpha=0.5) for color in anti_colors.values()]
        plt.legend(markers, anti_colors.keys(), numpoints=1, bbox_to_anchor=(1.04, 0.5), loc=2)
        plt.show()


def tsne_pca(smp, n=3, dim="both"):
    """ Visualizing 2d and 3d plot with the 2 or 3 TSNE components being feed with n principal components of a previous PCA.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        n (int): how many pca components should be used -> choose such that at least 90% of the variance is explained by them
        dim (str): 2d, 3d or both - for visualization
    """
    smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    x = smp_labelled.drop(["label", "smp_idx"], axis=1)
    y = smp_labelled["label"]
    anti_colors = {ANTI_LABELS[key] : value for key, value in COLORS.items() if key in smp_labelled["label"].unique()}
    pca = PCA(n_components=n, random_state=42)
    pca_result = pca.fit_transform(x)
    print("Cumulative explained variation for {} principal components: {}".format(n, np.sum(pca.explained_variance_ratio_)))

    if dim == "2d" or dim == "both":
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
        tsne_pca_results = tsne.fit_transform(pca_result)
        smp_pca_tsne = pd.DataFrame({"tsne-pca{}-one".format(n): tsne_pca_results[:, 0], "tsne-pca{}-two".format(n): tsne_pca_results[:, 1], "label": y})

        sns.scatterplot(x="tsne-pca{}-one".format(n), y="tsne-pca{}-two".format(n), hue="label", palette=COLORS, data=smp_pca_tsne, alpha=0.3)
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in anti_colors.values()]
        plt.legend(markers, anti_colors.keys(), numpoints=1, loc="center left", bbox_to_anchor=(1.04, 0.5))
        plt.show()

    if dim == "3d" or dim == "both":
        tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300, random_state=42)
        tsne_pca_results = tsne.fit_transform(pca_result)
        smp_pca_tsne = pd.DataFrame({"tsne-one": tsne_pca_results[:, 0], "tsne-two": tsne_pca_results[:, 1], "tsne-three": tsne_pca_results[:, 2], "label": y})

        ax = plt.figure(figsize=(16,10)).gca(projection="3d")
        color_labels = [COLORS[label] for label in smp_pca_tsne["label"]]
        ax.scatter(xs=smp_pca_tsne["tsne-one"], ys=smp_pca_tsne["tsne-two"], zs=smp_pca_tsne["tsne-three"], c=color_labels, alpha=0.3)
        ax.set_xlabel("tsne-one")
        ax.set_ylabel("tsne-two")
        ax.set_zlabel("tsne-three")
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in anti_colors.values()]
        plt.legend(markers, anti_colors.keys(), numpoints=1, bbox_to_anchor=(1.04, 0.5), loc=2)
        plt.show()

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

def visualize_tree(rf, x_train, y_train, tree_idx=0, min_samples_leaf=1000, file_name="tree"):
    """ Visualizes a single tree from a decision tree. Works only explicitly for my current data.
    Parameters:
        rf (RandomForestClassifier): the scikit learn random forest classfier
        x_train: Input data for training
        y_train: Target data for training
        tree_idx: Indicates which tree from the random forest should be visualized?
        min_samples_leaf: Indicates how many samples should be sorted to a leaf minimally
        file_name: The name under which the resulting png should be saved
    """
    # deciding directly which label gets which decision tree label
    y_train[y_train==6.0] = 0
    y_train[y_train==3.0] = 1
    y_train[y_train==5.0] = 2
    y_train[y_train==12.0] = 3
    y_train[y_train==4.0] = 4
    y_train[y_train==17.0] = 5
    y_train[y_train==16.0] = 6
    anti_labels = {0:"rgwp", 1:"dh", 2: "mfdh", 3:"dhwp", 4:"dhid", 5:"rare", 6:"pp"}

    # fit the model
    rf.fit(x_train, y_train)
    # extract one decision tree
    estimator = rf.estimators_[tree_idx]
    # we have to prune the tree otherwise the tree is way too big
    prune(estimator, min_samples_leaf=min_samples_leaf)
    class_names = [anti_labels[label] for label in y_train.unique()]
    # export image as dot file
    export_graphviz(estimator, out_file = file_name + ".dot",
                feature_names = x_train.columns,
                class_names = class_names,
                rounded = True, proportion = False,
                precision = 2, filled = True)
    # make a png file from the dot file and delete the dot file
    os.system("dot -Tpng "+ file_name + ".dot -o " + file_name + ".png")
    os.system("rm " + file_name + ".dot")

def all_in_one_plot(smp, show_indices=False, sort=True, title=None, file_name="plots/bogplots/all_in_one_labels.png"):
    """ Creates a plot where all profiles are visible with their labels.
    Plot can only be saved, not shown (GUI too slow for the plot).
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        show_indices (bool): if the SMP profile indices should be displayed
        sort (bool): if the SMP profiles should be sorted according to length
        title (str): Title of the plot
        file_name (str): where the resulting picture should be saved
    """
    #plt.rcParams.update({'font.size': 22})
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
    anti_colors = {ANTI_LABELS[key] : value for key, value in COLORS.items() if key in labelled_smp["label"].unique()}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in anti_colors.values()]
    plt.yticks(range(0, max_distance, 100))
    plt.legend(markers, anti_colors.keys(), numpoints=1, loc="upper left")#, markerscale=3)
    plt.ylabel("Distance from Ground [mm]")
    if title is None: title = "SMP Profiles with Labels"
    plt.title(title)

    if show_indices:
        labels = [str(int(smp_index)) for smp_index in smp_indices]
        plt.xticks(labels=labels, ticks=x_ticks, rotation=90, fontsize=8)
    else:
        plt.xticks([])

    plt.xlabel("SMP Profile Indices")
    plt.xlim(0.0, 1.0)
    plt.ylim(0, int(math.ceil(max_distance / 100.0)) * 100) # rounds up to next hundred
    figure = plt.gcf()  # get current figure
    #figure.set_size_inches(27.2, 15.2) # set size of figure
    #plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.savefig(file_name)
    plt.close()

def plot_confusion_matrix(confusion_matrix, labels, name="", save_file=None):
    """ Plot confusion matrix with relative prediction frequencies per label as heatmap.
    Parameters:
        confusion_matrix (nested list): 2d nested list with frequencies
        labels (list): list of tags or labels that should be used for the plot.
            Must be in the same order like the label order of the confusion matrix.
        name (str): Name of the model for the plot
        save_file (str): path where the plot should be saved. If None the plot is
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
                    annot_kws={"fontsize":7})
    # change font size of cbar axis
    #g.figure.axes[-1].yaxis.label.set_size(14)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label" + stats_text)
    plt.title("Confusion Matrix of {} Model \n".format(name))
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()

def plot_roc_curve(y_trues, y_prob_preds, labels, name="", save_file=None):
    """ Plotting ROC curves for all labels of a multiclass classification problem.
    Inspired from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    Parameters:
        y_trues
        y_prob_preds
        labels
        name
        save_file (str): path where the plot should be saved. If None the plot is
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
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()

def visualize_normalized_data(smp):
    """ Visualization after normalization and summing up classes has been achieved.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    # ATTENTION: don't print bogplots or single profiles! The results are just wrong after normalization!!!

    #smp_profile_name = "S31H0368"
    # HOW BALANCED IS THE LABELLED DATASET?
    #plot_balancing(smp)
    # SHOW THE DATADISTRIBUTION OF ALL FEATURES
    #pairwise_features(smp, features=["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], samples=200)
    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    #corr_heatmap(smp, labels=[3, 4, 5, 6, 12, 17])
    # Correlation does not help for categorical + continuous data - use ANOVA instead
    # FEATURE "EXTRACTION"
    #anova(smp, "plots/tables/ANOVA_results.txt", tablefmt="psql") # latex_raw also possible
    # RANDOM FOREST FEATURE EXTRACTION
    #forest_extractor(smp)
    # SHOW ONE SMP PROFILE WITHOUT LABELS
    #smp_unlabelled(smp, smp_name=smp_profile_name)
    # SHOW ONE SMP PROFILE WITH LABELS
    #smp_labelled(smp, smp_name=smp_profile_name)
    # PLOT ALL FEATURES AS LINES IN ONE PROFILE
    #smp_features(smp, smp_name=smp_profile_name, features=["mean_force", "var_force", "delta_4", "delta_12", "gradient"])

    # PLOT BOGPLOT
    # bog_plot(smp)
    # bog_label_plot(smp) # does not work correctly
    # smp_labelled(smp, smp_name=2000367.0)
    all_in_one_plot(smp, file_name="plots/bogplots/all_in_one_all_labels.png")

    # PCA and TSNE
    #pca(smp)
    # tsne(smp)
    #tsne_pca(smp, n=5)




def visualize_original_data(smp):
    """ Visualizing some things of the original data
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    # smp_profile_name = "S31H0368" #"S31H0607"
    # # HOW BALANCED IS THE LABELLED DATASET?
    # plot_balancing(smp)
    # # SHOW THE DATADISTRIBUTION OF ALL FEATURES
    # pairwise_features(smp, features=["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], samples=2000)
    # # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    # corr_heatmap(smp, labels=[3, 4, 5, 6, 7, 8, 9, 10])
    # # Correlation does not help for categorical + continuous data - use ANOVA instead
    # # FEATURE "EXTRACTION"
    # anova(smp, "plots/tables/ANOVA_results.txt", tablefmt="psql") # latex_raw also possible
    # # TODO: RANDOM FOREST FEATURE EXTRACTION
    # # SHOW ONE SMP PROFILE WITHOUT LABELS
    # smp_unlabelled(smp, smp_name=smp_profile_name)
    # # SHOW ONE SMP PROFILE WITH LABELS
    # smp_labelled(smp, smp_name=smp_profile_name)
    # # PLOT ALL FEATURES AS LINES IN ONE PROFILE
    # smp_features(smp, smp_name=smp_profile_name, features=["mean_force", "var_force", "delta_4", "delta_12", "gradient"])

    # PLOT BOGPLOT
    #bog_plot(smp)
    #bog_label_plot(smp) # does not work correctly
    #smp_labelled(smp, smp_name=2000367.0)
    all_in_one_plot(smp)

    # PCA and TSNE
    #pca(smp)
    #tsne(smp)
    #tsne_pca(smp, n=5)

def main():
    # load dataframe with smp data
    smp = load_data("smp_lambda_delta_gradient.npz")

    # visualize the original data
    visualize_original_data(smp)


if __name__ == "__main__":
    main()
