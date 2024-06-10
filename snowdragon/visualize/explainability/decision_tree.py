import os
import graphviz
import sklearn.tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tabulate import tabulate
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

from snowdragon import OUTPUT_DIR

# important setting to scale the pictures correctly
plt.rcParams.update({"figure.dpi": 250})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def forest_extractor(
        smp: pd.DataFrame, 
        file_name: Path = None, 
        tablefmt: str = "psql", 
        plot: bool = False
    ):
    """ Random Forest for feature extraction.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
        file_name (Path): in case the results should be saved in a file, indicate the path here
        tablefmt (str): table format that should be used for tabulate, e.g. 'psql' or 'latex_raw'
        plot (bool): shows a plotbar with the ranked feature importances
    """
    smp_labelled = smp[(smp["label"] != 0) & (smp["label"] != 2)]
    x = smp_labelled.drop(["label", "smp_idx"], axis=1)
    y = smp_labelled["label"]
    # Build a forest and compute the impurity-based feature importances
    forest = RandomForestClassifier(n_estimators=250, random_state=42)
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

def prune(
        decisiontree: sklearn.tree, 
        min_samples_leaf: int = 1
    ):
    """ Function for posterior decision tree pruning.
    Paramters:
        decisiontree (sklearn.tree): The decision tree to prune
        min_samples_leaf (int): How many samples should be sorted to this leaf minimally?
    """
    if decisiontree.min_samples_leaf >= min_samples_leaf:
        raise Exception("Tree already more pruned")
    else:
        decisiontree.min_samples_leaf = min_samples_leaf
        tree = decisiontree.tree_
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            if n_samples <= min_samples_leaf:
                tree.children_left[i]=-1
                tree.children_right[i]=-1

def visualize_tree(
        rf: RandomForestClassifier, 
        x_train: pd.DataFrame, 
        y_train: pd.Series, 
        anti_labels: dict,
        feature_names: list = None, 
        tree_idx: int = 0, 
        min_samples_leaf: int = 1000, 
        file_name: str = str(OUTPUT_DIR / "tree"), 
        format: str = "png"
    ):
    """ Visualizes a single tree from a decision tree. Works only explicitly for my current data.
    Parameters:
        rf (RandomForestClassifier): the scikit learn random forest classfier
        x_train (pd.DataFrame): Input data for training. If None the rf is pretrained.
        y_train (pd.Series): Target data for training. If None the rf is pretrained.
        anti_labels (dict): Label dictionary from configs, inversed. Matches the number identifier (int) to the string describing the grain: <int: str> 
        feature_names (list): Default None, since this is assigned from training data.
            If the rf is pretrained, this must be assigned here. (e.g. smp.columns)
        tree_idx (int): Indicates which tree from the random forest should be visualized?
        min_samples_leaf (int): Indicates how many samples should be sorted to a leaf minimally
        file_name (str): The name under which the resulting png should be saved (without extension!)
        format (str): e.g. png or svg, indicates how the pic should be stored
    """

    if (x_train is not None) and (y_train is not None):
        # deciding directly which label gets which decision tree label
        class_names = [anti_labels[int(label)] for label in y_train.unique()]

        feature_names = x_train.columns

        # fit the model
        rf.fit(x_train, y_train)

    else:
        feature_names = feature_names
        class_names = [anti_labels[c] for c in rf.classes_]

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