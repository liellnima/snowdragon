# TODO check imports
from data_handling.data_parameters import ANTI_LABELS, COLORS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# TODO separate between plotting and calculating
def pca(smp, n=3, dim="both", biplot=True):
    """ Visualizing 2d and 2d plot with the 2 or 3 principal components that explain the most variance.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        n (int): how many principal components should be extracted
        dim (str): 2d, 3d or both - for visualization
        biplot (bool): indicating if the features most used for the principal components should be plotted as biplot
    """
    plt.rcParams.update({"figure.dpi": 120})
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
    # print explained variance plot
    cum_vars = [sum(pca.explained_variance_ratio_[:(i+1)])*100 for i in range(len(pca.explained_variance_ratio_))]
    plt.ylabel("Explained Variance [%]")
    plt.xlabel("Number of Features")
    plt.title("Cumulatative Explained Variance of PCA Analysis")
    plt.ylim(30,100.5)
    plt.xlim(1, len(pca.explained_variance_ratio_))
    plt.grid()
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), cum_vars)
    plt.show()
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
        plt.legend(markers, anti_colors.keys(), numpoints=1, loc="upper right")#loc="center left", bbox_to_anchor=(1.04, 0.5))
        plt.title("PCA on all Labelled SMP Profiles (2-dim)")
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
        plt.title("PCA on all Labelled SMP Profiles (3-dim)")
        plt.show()

def tsne(smp, dim="both"):
    """ Visualizing 2d and 2d plot with the 2 or 3 TSNE components.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        dim (str): 2d, 3d or both - for visualization
    """
    plt.rcParams.update({"figure.dpi": 120})
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
        plt.legend(markers, anti_colors.keys(), numpoints=1, loc="upper right")#, bbox_to_anchor=(1.04, 0.5))
        plt.title("t-SNE on all Labelled SMP Profiles (2-dim)")
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
        plt.title("t-SNE on all Labelled SMP Profiles (3-dim)")
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
