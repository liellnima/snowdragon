# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_preprocessing import idx_to_int
from data_handling.data_parameters import LABELS

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pingouin as pg

from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

ANTI_LABELS = {0: "not_labelled",  1: "surface", 2: "ground", 3: "dh", 4: "dhid", 5: "mfdh", 6: "rgwp",
          7: "df", 8: "if", 9: "ifwp", 10:"sh", 11: "drift_end", 12: "snow-ice", 13: "dhwp", 14: "mfcl", 15: "mfsl", 16: "mfcr", 17: "pp"}

COLORS = {0: "lightsteelblue", 1: "chocolate", 2: "darkslategrey", 3: "lightseagreen", 4: "mediumaquamarine", 5: "midnightblue",
          6: "tomato", 7: "mediumvioletred", 8: "firebrick", 9: "lightgreen", 10: "orange", 11: "black", 12: "paleturquoise",
          13: "gold", 14: "orchid", 15: "fuchsia", 16: "brown", 17: "lila"}


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

def corr_heatmap(smp):
    """ Plots a correlation heatmap of all features.
    Paramters:
        smp (df.Dataframe): SMP preprocessed data
    """
    smp_corr = smp_filtered.corr()
    stats.pointbiserialr(["label"])
    mask = np.triu(np.ones_like(smp_corr, dtype=np.bool))
    mask = mask[1:, :-1]
    corr = smp_corr.iloc[1:, :-1].copy()
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f")
    plt.title("Correlation Heat Map of SMP Features with Label {}".format(ANTI_LABELS[label]))
    plt.show()

def anova(smp):
    """ Prints ANOVA F-scores for features.
    Paramters:
        smp (df.Dataframe): SMP preprocessed data
    """
    np.set_printoptions(precision=3)
    smp_filtered = smp[smp["label"] != 0]
    features = smp_filtered.drop("label", axis=1)
    target = smp_filtered["label"]
    # feature extraction
    test = SelectKBest(score_func=f_classif, k="all")
    fit = test.fit(features, target)
    results = pd.DataFrame({"Feature" : features.columns, "ANOVA-F-value" : fit.scores_, "P-value" : fit.pvalues_})
    print(results.sort_values(by=["ANOVA-F-value"], ascending=False).to_markdown())

def pairwise_features(smp, features, samples=None, kde=False):
    """ Produces a plot that shows the relation between all the feature given in the features list.
    Paramters:
        smp (df.DataFrame): SMP preprocessed data
        features (list): contains all features that should be displayed for pairwise comparison
        samples (int): Default None, how many samples should be drawn from the lablled dataset
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
    Paramters:
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

def visualize_original_data(smp):
    """ Visualizing some things of the original data
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    smp_profile_name = "S31H0368" #"S31H0607"
    # HOW BALANCED IS THE LABELLED DATASET?
    #plot_balancing(smp)
    # SHOW THE DATADISTRIBUTION OF ALL FEATURES
    #pairwise_features(smp, features=["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], samples=200)
    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    #corr_heatmap(smp, label=0)
    # Correlation does not help for categorical + continuous data - use ANOVA instead
    # FEATURE "EXTRACTION"
    anova(smp)
    # TODO: RANDOM FOREST FEATURE EXTRACTION
    # SHOW ONE SMP PROFILE WITHOUT LABELS
    #smp_unlabelled(smp, smp_name=smp_profile_name)
    # SHOW ONE SMP PROFILE WITH LABELS
    #smp_labelled(smp, smp_name=smp_profile_name)
    # PLOT ALL FEATURES AS LINES IN ONE PROFILE
    #smp_features(smp, smp_name=smp_profile_name, features=["mean_force", "var_force", "delta_4", "delta_12", "gradient"])

def main():
    # load dataframe with smp data
    smp = load_data("smp_lambda_delta_gradient.npz")
    # visualize the original data
    visualize_original_data(smp)


if __name__ == "__main__":
    main()
