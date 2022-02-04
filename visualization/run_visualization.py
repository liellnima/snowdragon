# imports
from data_handling.data_loader import load_data
from visualization.plot_data import bog_plot, all_in_one_plot
from visualization.plot_dim_reduction import pca, tsne, tsne_pca
from visualization.plot_profile import smp_unlabelled, smp_labelled, smp_features
from visualization.plot_data import plot_balancing, corr_heatmap, anova, forest_extractor, pairwise_features
from visualization.plot_results import plot_confusion_matrix, plot_model_comparison, prepare_evaluation_data

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


def visualize_normalized_data(smp):
    """ Visualization after normalization and summing up classes has been achieved.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    path = "output/plots_data/normalized/"
    # ATTENTION: don't use bogplots or single profiles after normalization!
    #plt.rcParams.update({"figure.dpi": 180})
    smp_profile_name = "S31H0368"
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
    smp_profile_name = "S31H0368" #"S31H0607"
    # HOW BALANCED IS THE LABELLED DATASET?
    #plot_balancing(smp, file_name="output/plots_data/original/class_balance.svg", title=None)
    # SHOW THE DATADISTRIBUTION OF ALL FEATURES
    pairwise_features(smp, features=["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], samples=2000, file_name=path+"pairwise_features.png")
    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    corr_heatmap(smp, labels=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], file_name=path+"corr_heatmap_all.png")
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
    #all_in_one_plot(smp, file_name="output/plots_data/original/overview_data_updatedaxis.png", profile_name=smp_profile_name, title=None)
    #all_in_one_plot(smp, file_name="output/plots_data/original/overview_data_indices.png", show_indices=True)

def visualize_results(all_scores, label_acc, label_prec):
    """ Visualizing results such as confusion matrix, roc curves and accuracy plots
    Parameters:
        all_scores (pd.DataFrame): dataframe containing scores of all models. stored as csv in output/scores
        label_acc (pd.DataFrame): label accuracies of each model. stored as csv in output/scores
        label_prec (pd.DataFrame): label precisions of each model. stored as csv in output/scores
    """
    # resort label_acc (so the models have the right grouping order)
    label_acc = label_acc.reindex([0, 1, 2, 3, 9, 10, 4, 5, 6, 7, 8, 11, 12, 13])
    label_prec = label_prec.reindex([0, 1, 2, 3, 9, 10, 4, 5, 6, 7, 8, 11, 12, 13])

    # visualize the accuracies and precisions of the different models
    # plot_model_comparison(
    #     label_acc, plot_rare=False,
    #     file_name="output/results/model_comparison_acc.png",
    #     metric_name="accuracy"
    # )
    # plot_model_comparison(
    #     label_prec, plot_rare=False,
    #     file_name="output/results/model_comparison_prec.png",
    #     metric_name="precision"
    # )

    # retrieve and summarize the data for the confusion matrices and the roc curves
    names, cf_matrices, label_orders = prepare_evaluation_data("output/evaluation")

    # plot confusion matrices
    plot_confusion_matrix(cf_matrices, label_orders, names, file_name="output/plots_results/confusion_matrixes.png")

    # plot roc curves
    # TODO

def main():
    args = parser.parse_args()

    ## VISUALIZE DATA ##
    # load dataframe with smp data
    smp = load_data("data/all_smp_profiles_updated.npz")
    smp_preprocessed = load_data("data/all_smp_profiles_updated_normalized.npz")

    # visualize the original data
    if args.original_data: visualize_original_data(smp)

    # visualize the normalized data
    if args.normalized_data: visualize_normalized_data(smp_preprocessed)

    ## VISUALIZE RESULTS ##
    # load dataframe with performance data
    all_scores = pd.read_csv("output/scores/all_scores.csv")
    label_acc = pd.read_csv("output/scores/acc_labels.csv")
    label_prec = pd.read_csv("output/scores/prec_labels.csv")

    if args.results: visualize_results(all_scores, label_acc, label_prec)

if __name__ == "__main__":
    main()
