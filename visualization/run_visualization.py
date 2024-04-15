from data_handling.data_loader import load_data
from data_handling.data_parameters import LABELS, EXAMPLE_SMP_NAME, SMP_ORIGINAL_NPZ, SMP_NORMALIZED_NPZ, SMP_PREPROCESSED_TXT, EVAL_LOC
from visualization.plot_data import bog_plot, all_in_one_plot
from visualization.plot_dim_reduction import pca, tsne, tsne_pca
from visualization.plot_profile import smp_unlabelled, smp_labelled, smp_features
from visualization.plot_data import plot_balancing, corr_heatmap, anova, forest_extractor, pairwise_features, visualize_tree
from visualization.plot_results import plot_confusion_matrix, plot_model_comparison, prepare_evaluation_data, plot_roc_auc, plot_test_bogplots, plot_model_comparison_bars, prepare_score_data

import pickle
import joblib
import argparse
import pandas as pd
import matplotlib.pyplot as plt
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


def visualize_normalized_data(smp):
    """ Visualization after normalization and summing up classes has been achieved.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    path = "output/plots_data/normalized/"
    # ATTENTION: don't use bogplots or single profiles after normalization!
    #plt.rcParams.update({"figure.dpi": 180})
    smp_profile_name = EXAMPLE_SMP_NAME
    # HOW BALANCED IS THE LABELLED DATASET?
    plot_balancing(smp, file_name=path+"class_balance_normalized.svg", title=None)
    # SHOW THE DATADISTRIBUTION OF ALL FEATURES
    pairwise_features(smp, features=["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], samples=200, file_name=path+"pairwise_features.png")
    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    corr_heatmap(smp, labels=[3, 4, 5, 6, 12, 16, 17], file_name=path+"corr_heatmap_all.png")
    # SHOW HEATMAP BETWEEN FEATURES
    corr_heatmap(smp, labels=None, file_name=path+"corr_heatmap_features.png")
    # Correlation does not help for categorical + continuous data - use ANOVA instead
    # FEATURE "EXTRACTION"
    anova(smp, file_name=path+"anova.txt", tablefmt="latex_raw") # latex_raw also possible
    # RANDOM FOREST FEATURE EXTRACTION
    forest_extractor(smp, file_name=path+"forest_features.txt", plot=False, tablefmt="latex_raw")
    # PLOT ALL NORMALIZED FEATURES AS LINES IN ONE PROFILE
    smp_features(smp, smp_name=smp_profile_name, features=["mean_force", "var_force", "min_force_4", "max_force_4", "L_12", "gradient"], file_name=path+smp_profile_name+"_features.png")
    plt.rcParams.update({"figure.dpi": 250})
    # PLOT BOGPLOT
    bog_plot(smp, file_name=path+"bog_plot.png")
    # PLOT ALL IN ONE PLOT
    all_in_one_plot(smp, title=None, file_name=path+"overview_data_norm.png", profile_name=smp_profile_name)

    # PCA and TSNE
    pca(smp, n=24, biplot=False, file_name=path)
    tsne(smp, file_name=path)

    tsne_pca(smp, n=5, file_name=path)

def visualize_original_data(smp):
    """ Visualizing some things of the original data
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    # clean smp data from nan values (not preprocessed yet)
    smp = smp.fillna(0)
    path = "output/plots_data/original/"
    smp_profile_name = EXAMPLE_SMP_NAME #"S31H0607"
    # HOW BALANCED IS THE LABELLED DATASET?
    #plot_balancing(smp, file_name="output/plots_data/original/class_balance.svg", title=None)
    # SHOW THE DATADISTRIBUTION OF ALL FEATURES
    pairwise_features(smp, features=["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], samples=2000, file_name=path+"pairwise_features.png")
    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    cleaned_labels = list(LABELS.values())
    cleaned_labels.remove(0) # remove not labelled
    cleaned_labels.remove(1) # remove surface
    cleaned_labels.remove(2) # remove ground
    corr_heatmap(smp, labels=cleaned_labels, file_name=path+"corr_heatmap_all.png")
    # Correlation does not help for categorical + continuous data - use ANOVA instead
    # FEATURE "EXTRACTION"
    anova(smp, path+"anova.txt", tablefmt="psql") # latex_raw also possible
    # TODO: RANDOM FOREST FEATURE EXTRACTION
    # SHOW ONE SMP PROFILE WITHOUT LABELS
    smp_unlabelled(smp, smp_name=smp_profile_name, file_name=path+smp_profile_name+"_unlabelled.png")
    # SHOW ONE SMP PROFILE WITH LABELS
    smp_labelled(smp, smp_name=smp_profile_name, file_name=path+smp_profile_name+"_labelled.png")
    # PLOT ALL FEATURES AS LINES IN ONE PROFILE
    smp_features(smp, smp_name=smp_profile_name, features=["mean_force", "var_force", "delta_4", "delta_12", "gradient"], file_name=path+smp_profile_name+"_features.png")

    # PLOT BOGPLOT
    bog_plot(smp, file_name=path+"bog_plot.png")
    # PLOT ALL IN ONE PLOT
    all_in_one_plot(smp, file_name="output/plots_data/original/overview_data_updatedaxis.png", profile_name=smp_profile_name, title=None)
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
    # resort label_acc (so the models have the right grouping order)
    label_acc = label_acc.reindex([7, 5, 2, 0, 4, 10, 6, 11, 8, 9, 12, 3, 1, 13])
    label_prec = label_prec.reindex([7, 5, 2, 0, 4, 10, 6, 11, 8, 9, 12, 3, 1, 13])
    #label_acc = label_acc.reindex([0, 1, 2, 3, 9, 10, 4, 5, 6, 7, 8, 11, 12, 13])
    #label_prec = label_prec.reindex([0, 1, 2, 3, 9, 10, 4, 5, 6, 7, 8, 11, 12, 13])

    # visualize the accuracies and precisions of the different models
    if comparison:
        plot_model_comparison_bars(
            label_acc, all_scores, plot_rare=False,
            file_name="output/plots_results/model_comparison_bar_acc.pdf",
            metric_name="accuracy"
        )
        plot_model_comparison_bars(
            label_prec, all_scores, plot_rare=False,
            file_name="output/plots_results/model_comparison_bar_prec.pdf",
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
            plot_confusion_matrix(cf_matrices_group,
                label_orders_group,
                names_group,
                file_name="output/plots_results/confusion_matrixes_" + str(i) + ".pdf")

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
            plot_roc_auc(y_group, y_pred_group,
                labels_group, names_group, legend=True,
                file_name="output/plots_results/roc_auc_curves.pdf")
        if bog_plot:

            # get smp indices for that
            # TODO move that to main function
            with open(SMP_PREPROCESSED_TXT, "rb") as myFile:
                smp_idx = pickle.load(myFile)["smp_idx_test"]

            # y_true chose anyone, all the same
            plot_test_bogplots(y_pred_group, y_group[0], smp_idx,
                labels_group, names_group,
                file_name="output/plots_results/bogplots_testset.pdf")




def main():
    # set this one to true when doing it the first time
    prepare_scores = False
    args = parser.parse_args()

    ## VISUALIZE DATA ##
    # load dataframe with smp data
    smp = load_data(SMP_ORIGINAL_NPZ)
    smp_preprocessed = load_data(SMP_NORMALIZED_NPZ)

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
        visualize_tree(rf, x_train=None, y_train=None, tree_idx=1,
            feature_names=feature_names,
            file_name="output/decision_tree" + "_" + str(tree_idx),
            min_samples_leaf=2500, format="svg")

    if args.tsne:
        tsne(smp_preprocessed, dim="2d", file_name="output/plots_data/normalized/tsne_2d_updated_")

    ## VISUALIZE RESULTS ##
    if args.results:
        # load dataframe with performance data
        if prepare_scores:
            prepare_score_data("output/evaluation/")
            #prepare_score_data("/home/julia/Documents/University/BA/Archive/evaluation_original_experiments/evaluation/")
        all_scores = pd.read_csv("output/scores/all_scores.csv")
        label_acc = pd.read_csv("output/scores/acc_labels.csv")
        label_prec = pd.read_csv("output/scores/prec_labels.csv")

        visualize_results(all_scores, label_acc, label_prec,
                          cf_matrix=True,
                          roc_auc=False,
                          bog_plot=False,
                          comparison=False)
    ## PREDICTIONS ##
    # TODO run this here instead of in plot_profile


if __name__ == "__main__":
    main()
