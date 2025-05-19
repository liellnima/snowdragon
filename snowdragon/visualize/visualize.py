import pickle
import joblib
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from snowdragon import OUTPUT_DIR
from snowdragon.utils.helper_funcs import load_smp_data

# unchecked
from snowdragon.visualize.data.dim_reduction import pca, tsne, tsne_pca
# checked
from snowdragon.visualize.data.all_profiles import bog_plot, all_in_one_plot
# checked
from snowdragon.visualize.data.profile import smp_unlabelled, smp_labelled, smp_features
# checked
from snowdragon.visualize.data.dataset import plot_balancing, corr_heatmap, anova, pairwise_features
# checked
from snowdragon.visualize.explainability.decision_tree import forest_extractor, visualize_tree
# unchecked
from snowdragon.visualize.results.plot_results import plot_confusion_matrix, prepare_evaluation_data, plot_roc_auc, plot_test_bogplots, plot_model_comparison_bars, prepare_score_data


# important setting to scale the pictures correctly
plt.rcParams.update({"figure.dpi": 250, "figure.figsize": (10, 5)})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# Example of how to use the visualization
# python -m visualization.run_visualization --original_data --normalized_data --results

# argparser
parser = argparse.ArgumentParser(description="Visualization of original data and results.")

parser.add_argument("--original_data", action="store_true", help="Plot visualizations for original data.")
parser.add_argument("--normalized_data", action="store_true", help="Plot visualizations for normalized data.")
parser.add_argument("--results", action="store_true", help="Plot visualizations for results.")
parser.add_argument("--tree", action="store_true", help="Plot decision tree.")
parser.add_argument("--tsne", action="store_true", help="Plot T-SNE.")


def visualize_normalized_data(
        smp: pd.DataFrame, 
        example_smp_name: str,
        example_smp_path: Path,
        used_labels: list,
        labels: dict,
        anti_labels: dict, 
        anti_labels_long: dict,
        colors: dict,
        plot_balanced_dataset: bool = True,
        plot_pairwise_features: bool = True,
        plot_correlation_heatmap_all_features: bool = True,
        plot_correlation_heatmap_between_features: bool = True,
        plot_anova: bool = True,
        plot_random_forest_feature_extraction: bool = True,
        plot_smp_features: bool = True, 
        plot_bog_plot: bool = True, 
        plot_all_in_one_plot: bool = True,
        plot_pca: bool = True,
        plot_tsne: bool = True,
        plot_tsne_pca: bool = True,
        **kwargs,
        ):
    """ Visualization after normalization and summing up classes has been achieved.
    Parameters:
        smp (pd.DataFrame): SMP preprocessed data
        TO BE ADDED
    """
    print("\tPlotting normalized data:")
    store_path = OUTPUT_DIR / "plots_data" / "normalized"
    # ATTENTION: don't use bogplots or single profiles after normalization!
    #plt.rcParams.update({"figure.dpi": 180})
    store_path.mkdir()

    # shows how balanced is the labelled dataset
    if plot_balanced_dataset:
        print("\t\tPlotting dataset balance ...")
        plot_balancing(
            smp, 
            colors = colors, 
            anti_labels = anti_labels, 
            anti_labels_long = anti_labels_long,
            file_name = store_path / "class_balance_normalized.svg", 
            title = None,
        )

    # show the data distribution of all features
    if plot_pairwise_features:
        print("\t\tPlotting pairwise features ...")
        pairwise_features(
            smp, 
            features = ["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], 
            anti_labels = anti_labels, 
            colors = colors,
            samples = 200, 
            file_name = store_path / "pairwise_features.png",
        )

    # show correlation heatmap of all features (with what are the labels correlated the most?)
    if plot_correlation_heatmap_all_features:
        print("\t\tPlotting correlation heatmaps of all features...")
        corr_heatmap(
            smp,
            labels = labels, 
            anti_labels = anti_labels, 
            correlation_labels = used_labels, 
            file_name = store_path / "corr_heatmap_all.png",
        )
    

    # show correlation heatmap between features
    if plot_correlation_heatmap_between_features:
        print("\t\tPlotting correlation heatmaps between all features...")
        corr_heatmap(
            smp, 
            labels = labels,
            anti_labels = anti_labels,
            correlation_labels = None, 
            file_name = store_path / "corr_heatmap_features.png",
        )

    # Correlation does not help for categorical + continuous data - use ANOVA instead
    if plot_anova:
        print("\t\tPlotting anova ...")
        anova(
            smp, 
            file_name = store_path / "anova.txt", 
            tablefmt = "psql", #"latex_raw" for publications
        )

    # random forest feature extraction
    if plot_random_forest_feature_extraction:
        print("\t\tPlotting decision tree feature importance ...")
        forest_extractor(
            smp, 
            file_name = store_path / "forest_features.txt", 
            plot = False, 
            tablefmt = "psql", # "latex_raw" for publications
        )

    # plot all normalized features as lines in one profile
    if plot_smp_features:
        print("\t\tPlotting features in SMP Profile ...")
        smp_features(
            smp, 
            smp_name = example_smp_name, 
            features = ["mean_force", "var_force", "min_force_4", "max_force_4", "L_12", "gradient"], 
            file_name = store_path / (str(example_smp_name) + "_features.png"),
        )
    
    plt.rcParams.update({"figure.dpi": 250})

    # plot bogplot
    if plot_bog_plot:
        print("\t\tPlotting bog plot ...")
        bog_plot(
            smp, 
            file_name = store_path / "bog_plot.png",
        )

    # plot all in one plot 
    if plot_all_in_one_plot:
        print("\t\tPlotting all-in-one-plot ...")
        all_in_one_plot(
            smp, 
            colors = colors,
            anti_labels_long = anti_labels_long,
            title = None, 
            file_name = store_path / "overview_data_norm.png", 
            profile_name = example_smp_name,
            example_smp_path = example_smp_path,
        )

    # pca 
    if plot_pca:
        print("\t\tPlotting pca ...")
        pca(
            smp, 
            colors = colors,
            anti_labels_long = anti_labels_long,
            n = 24, 
            biplot = False, 
            file_name = store_path,
        )
    
    # tsne
    if plot_tsne:
        print("\t\tPlotting tsne ...")
        tsne(
            smp, 
            colors = colors, 
            anti_labels_long = anti_labels_long,
            file_name = store_path,
        )

    # tsne and pca combined
    if plot_tsne_pca:
        print("\t\tPlotting tsne-pca ...")
        tsne_pca(
            smp, 
            colors = colors, 
            anti_labels_long = anti_labels_long,
            n = 5, 
            file_name = store_path,
        )

def visualize_original_data(
        smp: pd.DataFrame, 
        example_smp_name: str,
        example_smp_path: Path,
        labels: dict,
        anti_labels: dict, 
        anti_labels_long: dict,
        colors: dict,
        plot_balanced_dataset: bool = True,
        plot_pairwise_features: bool = True, 
        plot_correlation_heatmap: bool = True,
        plot_anova: bool = True, 
        plot_one_unlabelled_smp: bool = True,
        plot_one_labelled_smp: bool = True, 
        plot_smp_features: bool = True, 
        plot_bog_plot: bool = True, 
        plot_all_in_one_plot: bool = True,
        **kwargs,
        ):
    """ Visualizing some things of the original data
    Parameters:
        smp (pd.DataFrame): SMP preprocessed data
        TO BE ADDED
    """
    print("\tPlotting original data:")
    # clean smp data from nan values (not preprocessed yet)
    smp = smp.fillna(0)
    store_path = OUTPUT_DIR / "plots_data" / "original"
    store_path.mkdir()

    # show how balanced the labelled dataset is 
    if plot_balanced_dataset:
        print("\t\tPlotting dataset balance ...")
        plot_balancing(
            smp = smp, 
            colors = colors,
            anti_labels = anti_labels,
            anti_labels_long = anti_labels_long,
            file_name = store_path / "class_balance.svg", 
            title = None, 
        )

    # show the datadistribution of all features
    if plot_pairwise_features:
        print("\t\tPlotting pairwise features ...")
        pairwise_features(
            smp, 
            features = ["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], 
            anti_labels = anti_labels,
            colors = colors,
            samples = 2000, 
            file_name = store_path / "pairwise_features.png",
        )
        
    # show correlation heatmap of all features (with what are the labels correlated the most?)
    if plot_correlation_heatmap:
        print("\t\tPlotting correlation heatmaps...")
        cleaned_labels = list(labels.values())
        cleaned_labels.remove(0) # remove not labelled
        cleaned_labels.remove(1) # remove surface
        cleaned_labels.remove(2) # remove ground

        corr_heatmap(
            smp = smp, 
            labels = labels, 
            anti_labels = anti_labels,
            correlation_labels = cleaned_labels, 
            file_name = store_path / "corr_heatmap_all.png"
        )

    # Correlation does not help for categorical + continuous data - use ANOVA instead
    if plot_anova:
        print("\t\tPlotting anova ...")
        anova(
            smp, 
            file_name = store_path / "anova.txt", 
            tablefmt = "psql", # latex_raw also possible
        ) 

    # TODO: RANDOM FOREST FEATURE EXTRACTION

    # show one smp profile without labels
    if plot_one_unlabelled_smp:
        print("\t\tPlotting unlabelled SMP Profile ...")
        smp_unlabelled(
            smp, 
            smp_name = example_smp_name, 
            file_name = store_path / (str(example_smp_name) + "_unlabelled.png"),
        )

    # show one smp profile with lables
    if plot_one_labelled_smp:
        print("\t\tPlotting labelled SMP Profile ...")
        smp_labelled(
            smp, 
            colors = colors, 
            anti_labels = anti_labels,
            smp_name = example_smp_name, 
            file_name = store_path / (str(example_smp_name) + "_labelled.png"),
        )

    # plot all features as lines in one profile
    if plot_smp_features:
        print("\t\tPlotting features in SMP Profile ...")
        smp_features(
            smp, 
            smp_name = example_smp_name, 
            features = ["mean_force", "var_force", "delta_4", "delta_12", "gradient"], 
            file_name = store_path / (str(example_smp_name) + "_features.png"),
        )

    # plot bogplot
    if plot_bog_plot:
        print("\t\tPlotting bog plot ...")
        bog_plot(
            smp, 
            file_name = store_path / "bog_plot.png"
        )

    # plot all in one plot
    if plot_all_in_one_plot:
        print("\t\tPlotting all-in-one-plot ...")
        all_in_one_plot(
            smp, 
            colors = colors,
            anti_labels_long = anti_labels_long,
            file_name = store_path / "overview_data_updatedaxis.png", 
            profile_name = example_smp_name, 
            example_smp_path = example_smp_path,
            title = None,
        )
        #all_in_one_plot(smp, file_name="output/plots_data/original/overview_data_indices.png", show_indices=True)

def visualize_results(all_scores, label_acc, label_prec, cf_matrix=True, roc_auc=True, bog_plot=True, comparison=True):
    """ Visualizing results such as confusion matrix, roc curves and accuracy plots
    Parameters:
        all_scores (pd.DataFrame): dataframe containing scores of all models. stored as csv in output/scores
        label_acc (pd.DataFrame): label accuracies of each model. stored as csv in output/scores
        label_prec (pd.DataFrame): label precisions of each model. stored as csv in output/scores
        cf_matrix (bool): if confusion matrices should be created
        roc_auc (bool): if roc auc curves should be created
    """
    store_path = OUTPUT_DIR / "plots_results" 
    store_path.mkdir()

    # resort label_acc (so the models have the right grouping order)
    # TODO REMOVE or make this accessible to everyone
    label_acc = label_acc.reindex([7, 5, 2, 0, 4, 10, 6, 11, 8, 9, 12, 3, 1, 13])
    label_prec = label_prec.reindex([7, 5, 2, 0, 4, 10, 6, 11, 8, 9, 12, 3, 1, 13])

    # visualize the accuracies and precisions of the different models
    if comparison:
        plot_model_comparison_bars(
            label_acc, all_scores, plot_rare=False,
            file_name= store_path / "model_comparison_bar_acc.pdf",
            metric_name="accuracy"
        )
        plot_model_comparison_bars(
            label_prec, all_scores, plot_rare=False,
            file_name= store_path / "model_comparison_bar_prec.pdf",
            metric_name="precision"
        )

    # plot confusion matrices

    # retrieve and summarize the data for the confusion matrices and the roc curves
    names, cf_matrices, label_orders, y_trues, y_pred_probs = prepare_evaluation_data(EVAL_LOC)

    # plot cf matrices
    if cf_matrix:
        group_1 = ["baseline", "gmm", "bmm", "kmeans", "easy_ensemble", "knn"]
        group_2 = ["rf", "rf_bal", "svm", "lstm", "blstm", "enc_dec"]
        group_3 = ["self_trainer", "label_spreading"]

        for i, group in enumerate([group_1, group_2, group_3]):
            # get indices of the group and relevant names, matrices, etc.
            indices_group = [names.index(model) for model in group]
            names_group = []
            cf_matrices_group = []
            label_orders_group = []

            for idx in indices_group:
                names_group.append(names[idx])
                cf_matrices_group.append(cf_matrices[idx])
                label_orders_group.append(label_orders[idx])

            print(group)
            plot_confusion_matrix(
                cf_matrices_group,
                label_orders_group,
                names_group,
                file_name = store_path / "confusion_matrixes_" + str(i) + ".pdf")

    # plot roc curves
    if roc_auc or bog_plot:
        group = ["lstm", "rf", "self_trainer"]
        indices_group = [names.index(model) for model in group]
        y_group = []
        y_pred_group = []
        names_group = []
        labels_group = []

        for idx in indices_group:
            y_group.append(y_trues[idx])
            y_pred_group.append(y_pred_probs[idx])
            names_group.append(names[idx])
            labels_group.append(label_orders[idx])

        if roc_auc:
            plot_roc_auc(
                y_group, 
                y_pred_group,
                labels_group, 
                names_group, 
                legend=True,
                file_name= store_path / "roc_auc_curves.pdf",
            )

        if bog_plot:

            # get smp indices for that
            # TODO move that to main function
            with open(SMP_PREPROCESSED_TXT, "rb") as myFile:
                smp_idx = pickle.load(myFile)["smp_idx_test"]

            # y_true chose anyone, all the same
            plot_test_bogplots(
                y_pred_group, 
                y_group[0], 
                smp_idx,
                labels_group, 
                names_group,
                file_name= store_path / "bogplots_testset.pdf",
            )


def main():
    # set this one to true when doing it the first time
    prepare_scores = False
    args = parser.parse_args()

    ## VISUALIZE DATA ##
    # load dataframe with smp data
    smp = load_smp_data(SMP_ORIGINAL_NPZ)
    smp_preprocessed = load_smp_data(SMP_NORMALIZED_NPZ)

    # visualize the original data
    if args.original_data: visualize_original_data(smp)

    # visualize the normalized data
    if args.normalized_data: visualize_normalized_data(smp_preprocessed)

    if args.tree:
        # get random forest data
        with open("models/stored_models/rf.model", "rb") as handle:
            rf = joblib.load(handle)
        feature_names = list(smp.columns)
        feature_names.remove("label")
        feature_names.remove("smp_idx")
        tree_idx = 1
        visualize_tree(
            rf, 
            x_train=None, 
            y_train=None, 
            anti_labels=ANTI_LABELS,
            tree_idx=1,
            min_samples_leaf=2500, 
            feature_names=feature_names,
            file_name= str(OUTPUT_DIR / "decision_tree" + "_" + str(tree_idx)),
            format="svg"
        )

    if args.tsne:
        store_path = OUTPUT_DIR / "plots_data" / "normalized"
        store_path.mkdir()
        tsne(smp_preprocessed, dim="2d", file_name="output/plots_data/normalized/tsne_2d_updated_")

    ## VISUALIZE RESULTS ##
    if args.results:
        # load dataframe with performance data
        if prepare_scores:
            # TODO changes this to pathlib
            prepare_score_data("output/evaluation/")
        all_scores = pd.read_csv(OUTPUT_DIR / "scores" / "all_scores.csv")
        label_acc = pd.read_csv(OUTPUT_DIR / "scores" / "acc_labels.csv")
        label_prec = pd.read_csv(OUTPUT_DIR / "scores" / "prec_labels.csv")

        visualize_results(all_scores, label_acc, label_prec,
                          cf_matrix=True,
                          roc_auc=False,
                          bog_plot=False,
                          comparison=False)
    ## PREDICTIONS ##
    # TODO run this here instead of in plot_profile


if __name__ == "__main__":
    main()
