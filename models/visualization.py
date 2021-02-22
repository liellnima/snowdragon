# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_preprocessing import idx_to_int
from data_handling.data_parameters import LABELS, ANTI_LABELS, COLORS

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from scipy import stats
from tabulate import tabulate
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def smp_unlabelled(smp, smp_name):
    """ Plots a SMP profile without labels.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        smp_name (String): Name of the wished smp profile
    """
    smp_profile = smp[smp["smp_idx"] == idx_to_int(smp_name)]
    ax = sns.lineplot(smp_profile["distance"], smp_profile["mean_force"])
    plt.title("{} SMP Profile Distance (1mm layers) and Force".format(smp_name))
    ax.set_xlabel("Snow Depth [mm]")
    ax.set_ylabel("Mean Force [N]")
    plt.show()

def smp_labelled(smp, smp_name):
    """ Plots a SMP profile with labels.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        smp_name (String): Name of the wished smp profile
    """
    # SHOW THE SAME PROFILE WITH LABELS
    smp_profile = smp[smp["smp_idx"] == idx_to_int(smp_name)]
    ax = sns.lineplot(smp_profile["distance"], smp_profile["mean_force"])
    plt.title("{} SMP Profile Distance (1mm layers) and Force".format(smp_name))

    used_labels=[]
    last_label_num = 1
    last_distance = -1
    # going through labels and distance
    for label_num, distance in zip(smp_profile["label"], smp_profile["distance"]):
        if (label_num != last_label_num):
            # assign new background for each label
            background = ax.axvspan(last_distance, distance-1, color=COLORS[last_label_num], alpha=0.5)
            # set labels for legend
            if ANTI_LABELS[last_label_num] not in used_labels:
                background.set_label(ANTI_LABELS[last_label_num])
                used_labels.append(ANTI_LABELS[last_label_num])

            last_label_num = label_num
            last_distance = distance-1

        if distance == smp_profile.iloc[len(smp_profile)-1]["distance"]:
            ax.axvspan(last_distance, distance, color=COLORS[label_num], alpha=0.5).set_label(ANTI_LABELS[last_label_num])

    ax.legend()
    ax.set_xlabel("Snow Depth [mm]")
    ax.set_ylabel("Mean Force [N]")
    plt.show()

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
    # feature extraction
    test = SelectKBest(score_func=f_classif, k="all")
    fit = test.fit(features, target)
    results = pd.DataFrame({"Feature" : features.columns, "ANOVA-F-value" : fit.scores_, "P-value" : fit.pvalues_})
    results = results.sort_values(by=["ANOVA-F-value"], ascending=False)

    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(tabulate(results, headers='keys', tablefmt=tablefmt))

    print(tabulate(results, headers='keys', tablefmt=tablefmt))

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

# TODO why do depth hoar and depth hoar indurated not show up in the data? (only very few times)
def bog_label_plot(smp):
    """ Creates a bog plot for the given smp profiles. Makes the labels visible.
    Parameters:
        smp (pd.DataFrame): dataframe containing smp profiles
    """
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    anti_colors = {ANTI_LABELS[key] : value for key, value in COLORS.items() if key in labelled_smp["label"].unique()}
    my_colors = {key : value for key, value in COLORS.items() if key in labelled_smp["label"].unique()}
    distance_between_smp = 0.5
    day_id = 1
    smp_indices = labelled_smp["smp_idx"].unique()

    for i, curr_smp_idx in zip(range(len(smp_indices)), smp_indices):
        smp_profile = labelled_smp[labelled_smp["smp_idx"] == curr_smp_idx]
        Y = smp_profile["label"]
        z = smp_profile["distance"]
        x1 = i * distance_between_smp
        x2 = (i+1) * distance_between_smp
        colors_list = [value for key, value in my_colors.items() if key >= min(Y) and key <= max(Y)]
        plt.contourf([x1, x2], z, np.array([Y,Y]).transpose(), cmap=colors.ListedColormap(colors_list))

    plt.xlabel('Snow Micro Pen Profiles')
    plt.ylabel('Depth (mm)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in anti_colors.values()]
    plt.legend(markers, anti_colors.keys(), numpoints=1)
    plt.title("All Labelled SMP Profiles with Assigned Labels")
    plt.grid()
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

# TODO Labels and colors for 3d plot
# TODO cleaning up
def pca(smp):
    """ Visualizing 2d and 2d plot with the 2 or 3 principal components that explain the most variance.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    x = smp_labelled.drop(["label", "smp_idx"], axis=1)
    y = smp_labelled["label"]
    # first two components explain the most anyway!
    pca = PCA(n_components=3, random_state=42)
    pca_result = pca.fit_transform(x)
    smp_with_pca = pd.DataFrame({"pca-one": pca_result[:,0], "pca-two": pca_result[:,1], "pca-three": pca_result[:,2], "label": y})
    print("Explained variation per principal component: {}.".format(pca.explained_variance_ratio_))

    # 2d plot
    sns.scatterplot(x="pca-one", y="pca-two", hue="label", palette=COLORS, data=smp_with_pca, alpha=0.3)
    plt.show()

    # 3d plot
    ax = plt.figure(figsize=(16,10)).gca(projection="3d")
    ax.scatter(xs=smp_with_pca["pca-one"], ys=smp_with_pca["pca-two"], zs=smp_with_pca["pca-three"], c=smp_with_pca["label"])
    ax.set_xlabel("pca-one")
    ax.set_ylabel("pca-two")
    ax.set_zlabel("pca-three")
    plt.show()

# TODO Labels and colors for 3d plot
# TODO cleaning up
def tsne(smp):
    """ Visualizing 2d and 2d plot with the 2 or 3 TSNE components.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    x = smp_labelled.drop(["label", "smp_idx"], axis=1)
    y = smp_labelled["label"]
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(x)
    smp_with_tsne = pd.DataFrame({"tsne-one": tsne_results[:, 0], "tsne-two": tsne_results[:, 1], "tsne-three": tsne_results[:, 2], "label": y})

    # 2d plot
    sns.scatterplot(x="tsne-one", y="tsne-two", hue="label", palette=COLORS, data=smp_with_tsne, alpha=0.3)
    plt.show()

    # 3d plot
    ax = plt.figure(figsize=(16,10)).gca(projection="3d")
    ax.scatter(xs=smp_with_tsne["tsne-one"], ys=smp_with_tsne["tsne-two"], zs=smp_with_tsne["tsne-three"], c=smp_with_tsne["label"])
    ax.set_xlabel("tsne-one")
    ax.set_ylabel("tsne-two")
    ax.set_zlabel("tsne-three")
    plt.show()

# TODO clean-up, 3d plot, labeling, etc.
def tsne_pca(smp):
    """ Visualizing 2d and 2d plot with the 2 or 3 TSNE components being feed with 3 principal components of a previous PCA.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    x = smp_labelled.drop(["label", "smp_idx"], axis=1)
    y = smp_labelled["label"]
    pca = PCA(n_components=3, random_state=42)
    pca_result = pca.fit_transform(x)
    print("Cumulative explained variation for 3 principal components: {}".format(np.sum(pca.explained_variance_ratio_)))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result)
    smp_pca_tsne = pd.DataFrame({"tsne-pca3-one": tsne_pca_results[:, 0], "tsne-pca3-two": tsne_pca_results[:, 1], "label": y})
    sns.scatterplot(x="tsne-pca3-one", y="tsne-pca3-two", hue="label", palette=COLORS, data=smp_pca_tsne, alpha=0.3)
    plt.show()

# TODO cleanup, etc.
# save table, use different table format
# maybe add plot to ANOVA
def forest_extractor(smp):
    """ Random Forest for feature extraction.
    """
    smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    x = smp_labelled.drop(["label", "smp_idx"], axis=1)
    y = smp_labelled["label"]
    # Build a forest and compute the impurity-based feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=42)
    forest.fit(x, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    indices_names = [list(x.columns)[index] for index in indices]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, indices_names[f], importances[indices[f]]))
    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), importances[indices],
            color="lightgreen", yerr=std[indices], align="center")
    plt.xticks(range(x.shape[1]), indices_names, rotation=55)
    plt.xlim([-1, x.shape[1]])
    plt.show()

def visualize_normalized_data(smp):
    """ Visualization after normalization and summing up classes has been achieved.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    smp_profile_name = "S31H0368"
    # HOW BALANCED IS THE LABELLED DATASET?
    #plot_balancing(smp)
    # SHOW THE DATADISTRIBUTION OF ALL FEATURES
    #pairwise_features(smp, features=["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], samples=200)
    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    corr_heatmap(smp, labels=[3, 4, 5, 6, 12, 17])
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
    bog_label_plot(smp)

    # PCA and TSNE
    # pca(smp)
    # tsne(smp)
    # tsne_pca(smp)




def visualize_original_data(smp):
    """ Visualizing some things of the original data
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    # smp_profile_name = "S31H0368" #"S31H0607"
    # # HOW BALANCED IS THE LABELLED DATASET?
    #plot_balancing(smp)
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
    bog_label_plot(smp)

def main():
    # load dataframe with smp data
    smp = load_data("smp_lambda_delta_gradient.npz")

    # visualize the original data
    visualize_original_data(smp)


if __name__ == "__main__":
    main()
