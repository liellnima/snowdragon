from models.cv_handler import semisupervised_cv, assign_clusters

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, balanced_accuracy_score

# TODO make name a parameter and return_train_score as well

# ATTENTION: log_loss and roc_auc or other probability based metrics cannot be calculated for kmeans (not well defined!)
# https://towardsdatascience.com/cluster-then-predict-for-classification-tasks-142fdfdc87d6
def kmeans(unlabelled_data, x_train, y_train, cv, num_clusters=5, find_num_clusters="both", plot=True):
    """ Semisupervised kmeans algorithm. Assigns most frequent snow label to cluster.
    Parameters:
        unlabelled_data: Data on which the clustering should take place
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        num_clusters (int): number of clusters for kmeans, or maximum number of clusters
        find_num_clusters (str): either "sil" for Silhouette Coefficient or "acc" for balanced accuracy or "both".
            In case of "both" the optimal number of cluster is choosen according to results of acc
            Default: None - in this case only the kmeans model with num_clusters cluster is run
        plot (bool): whether silhouette coefficient or balanced accuracy should be plot
    Returns:
        dict: results from cross validation
    """
    if find_num_clusters is not None:
        max_cluster = num_clusters
        all_sil_scores = []
        bal_acc_scores = []
        # iterate through all possible numbers of clusters and calculate their sil score
        print("Find best number of clusters for kmeans:")
        for cluster_num in tqdm(range(2, max_cluster+1)):
            km = KMeans(n_clusters=cluster_num, init="random", n_init=cluster_num, random_state=42).fit(x_train)
            # calculate sil scores
            if find_num_clusters == "sil" or find_num_clusters == "both":
                sil_scores = silhouette_score(x_train, km.labels_, metric="euclidean")
                all_sil_scores.append(sil_scores)
            # calculate balanced accuracy
            if find_num_clusters == "acc" or find_num_clusters == "both":
                clusters = km.predict(x_train)
                y_pred = assign_clusters(y_train, clusters, cluster_num)
                bal_acc_scores.append(balanced_accuracy_score(y_train, y_pred))

        # find the argmax of the scores -> this is the perfect number of clusters
        # argmax of sil scores
        if find_num_clusters == "sil" or find_num_clusters == "both":
            sil_cluster_num_optimal = max(range(len(all_sil_scores)), key=lambda i: all_sil_scores[i]) + 2
            # the number of cluster which should be used: the optimal number of cluster
            # in case of "both" this will be overwritten by the balanced_acc maximum
            num_clusters = sil_cluster_num_optimal
        # argmax of bal_acc scores
        if find_num_clusters == "acc" or find_num_clusters == "both":
            acc_cluster_num_optimal = max(range(len(bal_acc_scores)), key=lambda i: bal_acc_scores[i]) + 2
            # the number of cluster which should be used: the optimal number of cluster
            num_clusters = acc_cluster_num_optimal

        # plot sil coefficients and balanced accuracy scores
        if plot:
            # plot for silhouette coefficient
            if find_num_clusters == "sil" or find_num_clusters == "both":
                plt.plot(range(2, max_cluster+1), all_sil_scores, label="Silhouette Coef")
                plt.axvline(sil_cluster_num_optimal, color="red", linestyle="--")
                plt.title("Silhouette Coefficient for K-means Clustering Model")
                plt.xlabel("Number of Clusters")
                plt.ylabel("Mean Silhouette Coefficient")
                plt.show()
            # plot for accuracy
            if find_num_clusters == "acc" or find_num_clusters == "both":
                plt.plot(range(2, max_cluster+1), bal_acc_scores, label="Balanced Acc")
                plt.axvline(acc_cluster_num_optimal, color="red", linestyle="--")
                plt.title("Balanced Accuracy for K-means Clustering Model")
                plt.xlabel("Number of Clusters")
                plt.ylabel("Balanced Accuracy")
                plt.show()

    km = KMeans(n_clusters=num_clusters, init="random", n_init=num_clusters, random_state=42)

    return semisupervised_cv(km, unlabelled_data, x_train, y_train, num_clusters, cv, name="Kmeans")

def gaussian_mix(unlabelled_data, x_train, y_train, cv, cov_type="tied", num_components=15, find_num_components="both", plot=True):
    """ Semisupervised Gaussian Mixture Algorithm. Assigns most frequent snow label to gaussians.
    Parameters:
        unlabelled_data: Data on which the clustering should take place
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        cov_type (str): type of covariance used for gaussian mixture model - one of: "tied", "diag", "spherical", "full"
        num_components (int): number of distributions (maximally) used for the model
        find_num_components (str): either "bic" for Bayesian Information Criterion or "acc" for balanced accuracy or "both".
            In case of "both" the optimal number of cluster is choosen according to results of acc
            Default: None - in this case only the kmeans model with num_clusters cluster is run
        plot (bool): whether the bic and balanced accuracy should be plot
    Returns:
        dict: results of crossvalidation
    """
    if find_num_components:
        max_components = num_components
        all_bic_scores = []
        bal_acc_scores = []

        print("Find best number of components for Gaussian Mixture Model:")
        for n_gaussians in tqdm(range(1, max_components+1)):
            gm = GaussianMixture(n_components=n_gaussians, init_params="random", max_iter=150, covariance_type=cov_type, random_state=42)
            gm.fit(x_train)
            # calculate bic score
            if find_num_components == "bic" or find_num_components == "both":
                all_bic_scores.append(gm.bic(x_train))
            # calculate balanced accuracy
            if find_num_components == "acc" or find_num_components == "both":
                clusters = gm.predict(x_train)
                y_pred = assign_clusters(y_train, clusters, n_gaussians)
                bal_acc_scores.append(balanced_accuracy_score(y_train, y_pred))

        # optimal number of distributions is the one with the lowest bayesian information criterion or highest accuracy
        if find_num_components == "bic" or find_num_components == "both":
            bic_components_num_optimal = min(range(len(all_bic_scores)), key=lambda i: all_bic_scores[i]) + 1
            # in case of "both" this will be overwritten by the balanced_acc maximum
            n_components = bic_components_num_optimal
        # argmax of bal_acc scores
        if find_num_components == "acc" or find_num_components == "both":
            acc_components_num_optimal = max(range(len(bal_acc_scores)), key=lambda i: bal_acc_scores[i]) + 1
            n_components = acc_components_num_optimal

        if plot:
            if find_num_components == "bic" or find_num_components == "both":
                plt.plot(range(1, max_components+1), all_bic_scores)
                plt.axvline(bic_components_num_optimal, color="red", linestyle="--")
                plt.title("Bayesian Information Criterion for Gaussian Mixture Model, {}".format(cov_type))
                plt.xlabel("Number of Gaussian Distributions")
                plt.ylabel("BIC")
                plt.show()
            if find_num_components == "acc" or find_num_components == "both":
                plt.plot(range(1, max_components+1), bal_acc_scores)
                plt.axvline(acc_components_num_optimal, color="red", linestyle="--")
                plt.title("Balanced Accuracy for Gaussian Mixture Model, {}".format(cov_type))
                plt.xlabel("Number of Gaussian Distributions")
                plt.ylabel("Balanced Accuracy")
                plt.show()

    gm = GaussianMixture(n_components=n_components, init_params="random", max_iter=150, covariance_type=cov_type, random_state=42)
    return semisupervised_cv(gm, unlabelled_data, x_train, y_train, n_gaussians, cv, name="GaussianMixture_{}".format(cov_type))

def bayesian_gaussian_mix(unlabelled_data, x_train, y_train, cv, cov_type="tied", num_components=15):
    """ Semisupervised Variational Bayesian estimation of a Gaussian Mixture Algorithm. Assigns most frequent snow label to gaussians.
    Find automatically the right number of
    Parameters:
        unlabelled_data: Data on which the clustering should take place
        x_train: Input data for training
        y_train: Target data for training
        cv (list of tuples): cross validation indices
        cov_type (str): type of covariance used for gaussian mixture model - one of: "tied", "diag", "spherical", "full"
        num_components (int): number of distributions maximally used for the model
    Returns:
        dict: results of crossvalidation
    """
    bgm = GaussianMixture(n_components=num_components, init_params="random", max_iter=150, covariance_type=cov_type, random_state=42)

    return semisupervised_cv(bgm, unlabelled_data, x_train, y_train, num_components, cv, name="BayesianGaussianMixture_{}".format(cov_type))


# TODO Delete after using the visualization
def kmeans_old():

    # k-means clustering for one sample
    km = KMeans(n_clusters=5, init="random", n_init=10, random_state=42)

    clusters = km.fit_predict(sample[["mean_force", "var_force"]])
    print(clusters)

    sns.scatterplot(sample["var_force"], sample["mean_force"], hue=sample["label"], style=clusters).set_title("Clustering of S31H0369")
    plt.show()

    # we have in total 10 labels
    km_more = KMeans(n_clusters=10, init="random", n_init=100, random_state=42)

    clusters = km_more.fit_predict(smp_more[["mean_force", "var_force"]])
    print(clusters)

    sns.scatterplot(smp_more["var_force"], smp_more["mean_force"], hue=clusters).set_title("Variance and Mean force of 1000 samples")
    plt.show()

    # k-means clustering for all which are labelled

    # we have in total 10 labels
    km_lab = KMeans(n_clusters=10, init="random", n_init=100, random_state=42)

    clusters = km_lab.fit_predict(smp_labelled[["mean_force", "var_force"]])
    print(clusters)

    sns.scatterplot(smp_labelled["var_force"], smp_labelled["mean_force"], hue=smp_labelled["label"], style=clusters).set_title("Clustering for labelled data")
    plt.show()
