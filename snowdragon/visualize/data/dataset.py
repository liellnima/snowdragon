import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tabulate import tabulate
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

from snowdragon import OUTPUT_DIR

# important setting to scale the pictures correctly
plt.rcParams.update({"figure.dpi": 250})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def plot_balancing(
        smp: pd.DataFrame, 
        colors: dict,
        anti_labels: dict,
        anti_labels_long: dict,
        title: str = "Distribution of Labels in the Labelled SMP Dataset",
        file_name: Path = OUTPUT_DIR / "plots_data" / "balance_of_dataset.svg",
    ):
    """ Produces a plot that shows how balanced the dataset is.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        colors (dict): Color dictionary from configs. Matches a grain number (int) to a color: < grain(int): color(str)>
        anti_labels (dict): Label dictionary from configs, inversed. Matches the number identifier (int) to the string describing the grain: <int: str> 
        anti_labels_long (dict): Label dictionary from configs, inversed, with  the grain types fully written out. Matches the int identifier to a string describing the grain: <int: str>
        title (str): Title. No title if None.
        file_name (Path): If None - plot is shown. If Path, the file is saved there.
    """
    # take only labelled data and exclude surface and ground
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]
    # I can currently not find snow-ice because I am cutting of the last datapoints, if they are less than 1mm
    print("Can I find the snow-ice label?", smp[smp["label"] == 11])

    my_colors = [colors[label_number] for label_number in labelled_smp["label"].value_counts().index]
    ax = sns.countplot(x="label", data=labelled_smp, order=labelled_smp["label"].value_counts().index, palette=my_colors)
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
    x_labels = [anti_labels[label_number] for label_number in labelled_smp["label"].value_counts().index]
    ax.set_xticklabels(x_labels, rotation=0)
    ax2.set_ylim(0,50)
    ax.set_ylim(0,len(labelled_smp)*0.5)

    # legend
    anti_colors = {anti_labels_long[key] : value for key, value in colors.items() if key in labelled_smp["label"].unique()}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', alpha=1) for color in anti_colors.values()]
    plt.legend(markers, anti_colors.keys(), numpoints=1, loc="upper right", fontsize=8)

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()

# Longterm TODO: more beautiful heatmaps: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
def corr_heatmap(
        smp: pd.DataFrame, 
        labels: dict,
        anti_labels: dict,
        correlation_labels: list = None, 
        file_name: Path = OUTPUT_DIR / "plots_data" / "corr_heatmap.png", 
        title : str = "",
    ):
    """ Plots a correlation heatmap of all features.
    Parameters:
        smp (df.Dataframe): SMP preprocessed data
        labels (dict): Label dictionary from configs. Matches a label (str) to a number (int): <str: int>
        anti_labels (dict): Label dictionary from configs, inversed. Matches the number identifier (int) to the string describing the grain: <int: str> 
        correlation_labels (list): Default None - usual complete correlation heatmap is calculated.
            Else put in the labels for which the correlation heatmap should be calculated
        file_name (Path): where the resulting pic should be saved
        title (str): title of the figure
    """
    if correlation_labels is None:
        smp_filtered = smp.drop("label", axis=1)
        smp_corr = smp_filtered.corr()
        mask = np.triu(np.ones_like(smp_corr, dtype=np.bool_))
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
        for corr_label in correlation_labels:
            # make the label integer if it not already
            if not isinstance(corr_label, int): corr_label = labels[corr_label]
            # add the label to smp_labelled
            col_name = anti_labels[corr_label]
            col_names.append(col_name)
            smp_labelled[col_name] = (smp_labelled["label"] == corr_label) * 1
        # drop label columns
        smp_labelled = smp_labelled.drop("label", axis=1)
        # calculate the correlation heatmap
        smp_corr = smp_labelled.corr() # Pearson Correlation
        # consider only the correlations between labels and features
        corr = smp_corr.iloc[-len(correlation_labels):, :].copy()
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


def anova(
        smp: pd.DataFrame, 
        file_name: Path = None, 
        tablefmt: str = "psql"
    ):
    """ Prints ANOVA F-scores for features.
    Parameters:
        smp (pd.Dataframe): SMP preprocessed data
        file_name (Path): in case the results should be saved in a file, indicate the path here
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
            f.write(tabulate(results, headers="keys", tablefmt=tablefmt))

    print("ANOVA Feature Ranking")
    print(tabulate(results, headers="keys", tablefmt=tablefmt))

def pairwise_features(
        smp: pd.DataFrame, 
        features: list, 
        anti_labels: dict, 
        colors: dict,
        samples: int = None, 
        kde: bool = False, 
        file_name: Path = OUTPUT_DIR / "plots_data" / "pairwise_features.png",
    ):
    """ Produces a plot that shows the relation between all the feature given in the features list.
    Parameters:
        smp (pd.DataFrame): SMP preprocessed data
        features (list): contains all features that should be displayed for pairwise comparison
        anti_labels (dict): Label dictionary from configs, inversed. Matches the number identifier (int) to the string describing the grain: <int: str> 
        colors (dict): Color dictionary from configs. Matches a grain number (int) to a color: < grain(int): color(str)>
        samples (int): Default None, how many samples should be drawn from the labelled dataset
        kde (bool): Default False, whether the lower triangle should overlay kde plots
        file_name (Path): where the pic should be saved. If 'None' the plot is shown.
    """
    # use only data that is already labelled
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]
    smp_filtered = labelled_smp[features] if samples is None else labelled_smp[features].sample(n=samples, random_state=42)
    g = sns.pairplot(smp_filtered, hue="label", palette=colors, plot_kws={"alpha": 0.5, "linewidth": 0})
    if kde : g.map_lower(sns.kdeplot, levels=4, color=".2")
    new_labels = [anti_labels[int(float(text.get_text()))] for text in g._legend.texts]
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    if file_name is not None:
        plt.savefig(file_name, dpi=200)
        plt.close()
    else:
        plt.show()
