# imports
from visualization.plot_data import plot_balancing, corr_heatmap, anova, forest_extractor, pairwise_features
from visualization.plot_data import bog_plot, all_in_one_plot,
from visualization.plot_dim_reduction import pca, tsne, tsne_pca
from visualization.plot_profile import smp_unlabelled, smp_labelled, smp_features

import matplotlib.pyplot as plt
# important setting to scale the pictures correctly
plt.rcParams.update({"figure.dpi": 250})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def visualize_normalized_data(smp):
    """ Visualization after normalization and summing up classes has been achieved.
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    # ATTENTION: don't print bogplots or single profiles! The results are just wrong after normalization!!!
    plt.rcParams.update({"figure.dpi": 180})
    smp_profile_name = "S31H0368"
    # HOW BALANCED IS THE LABELLED DATASET?
    plot_balancing(smp, file_name="plots/data_visual/normalized/class_balance_normalized.svg", title=None)
    # SHOW THE DATADISTRIBUTION OF ALL FEATURES
    #pairwise_features(smp, features=["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], samples=200)
    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    corr_heatmap(smp, labels=[3, 4, 5, 6, 12, 16, 17])
    # SHOW HEATMAP BETWEEN FEATURES
    corr_heatmap(smp, labels=None)
    # Correlation does not help for categorical + continuous data - use ANOVA instead
    # FEATURE "EXTRACTION"
    anova(smp, "plots/data_preprocessed/anova.txt", tablefmt="latex_raw") # latex_raw also possible
    # RANDOM FOREST FEATURE EXTRACTION
    forest_extractor(smp, file_name="plots/data_preprocessed/forest_features.txt", plot=True, tablefmt="latex_raw")
    # SHOW ONE SMP PROFILE WITHOUT LABELS
    #smp_unlabelled(smp, smp_name=smp_profile_name)
    # SHOW ONE SMP PROFILE WITH LABELS
    #smp_labelled(smp, smp_name=smp_profile_name)
    # PLOT ALL FEATURES AS LINES IN ONE PROFILE
    smp_features(smp, smp_name=smp_profile_name, features=["mean_force", "var_force", "min_force_4", "max_force_4", "L_12", "gradient"])
    plt.rcParams.update({"figure.dpi": 250})
    # PLOT BOGPLOT
    bog_plot(smp)
    # smp_labelled(smp, smp_name=2000367.0)
    all_in_one_plot(smp, title="Summarized Labels of all SMP Profiles", file_name="plots/data_preprocessed/bogplot_labels_normalized.png")

    # PCA and TSNE
    pca(smp, n=24, biplot=False)
    tsne(smp)

    #tsne_pca(smp, n=5)

def visualize_original_data(smp):
    """ Visualizing some things of the original data
    Parameters:
        smp (df.DataFrame): SMP preprocessed data
    """
    smp_profile_name = "S31H0368" #"S31H0607"
    # HOW BALANCED IS THE LABELLED DATASET?
    #plot_balancing(smp, file_name="plots/data_visual/original/class_balance.svg", title=None)
    # SHOW THE DATADISTRIBUTION OF ALL FEATURES
    #pairwise_features(smp, features=["label", "distance", "var_force", "mean_force", "delta_4", "lambda_4", "gradient"], samples=2000)
    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    #corr_heatmap(smp, labels=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    # Correlation does not help for categorical + continuous data - use ANOVA instead
    # FEATURE "EXTRACTION"
    #anova(smp, "plots/tables/ANOVA_results.txt", tablefmt="psql") # latex_raw also possible
    # TODO: RANDOM FOREST FEATURE EXTRACTION
    # SHOW ONE SMP PROFILE WITHOUT LABELS
    #smp_unlabelled(smp, smp_name=smp_profile_name)
    # SHOW ONE SMP PROFILE WITH LABELS
    #smp_labelled(smp, smp_name=smp_profile_name)
    # PLOT ALL FEATURES AS LINES IN ONE PROFILE
    #smp_features(smp, smp_name=smp_profile_name, features=["mean_force", "var_force", "delta_4", "delta_12", "gradient"])

    # PLOT BOGPLOT
    #bog_plot(smp, file_name=None)
    #smp_labelled(smp, smp_name=2000367.0)
    all_in_one_plot(smp, file_name="plots/data_visual/original/overview_data.svg", profile=smp_profile_name)

    #all_in_one_plot(smp, file_name="plots/data_visual/original/overview_data_indices.png", show_indices=True)

def main():
    # load dataframe with smp data
    smp = load_data("data/all_smp_profiles_updated.npz")

    # visualize the original data
    visualize_original_data(smp)


if __name__ == "__main__":
    main()
