# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_preprocessing import idx_to_int
from data_handling.data_parameters import LABELS
from models.visualization import visualize_original_data # TODO or something like this

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# TODO just import the metrics you need or everything
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# from sklearn.multioutput import MultiOutputClassifier



def kmeans():

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

def my_train_test_split(smp, test_size=0.2, train_size=0.8):
    """ Splits data into training and testing data
    Parameters:
        smp (df.DataFrame): Preprocessed smp data
        test_size (float): between 0 and 1, size of testing data
        train_size (float): between 0 and 1, size of training data
    Returns:
        quadruple: x_train, x_test, y_train, y_test
    """
    # labelled data
    labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]

    # print how many labelled profiles we have
    num_profiles = labelled_smp["smp_idx"].nunique()
    num_points = labelled_smp["smp_idx"].count()
    idx_list = labelled_smp["smp_idx"].unique()

    # sample randomly from the list
    train_idx, test_idx = train_test_split(idx_list, test_size=test_size, train_size=train_size, random_state=42)
    train = labelled_smp[labelled_smp["smp_idx"].isin(train_idx)]
    test = labelled_smp[labelled_smp["smp_idx"].isin(test_idx)]
    x_train = train.drop(["label"], axis=1)
    x_test = test.drop(["label"], axis=1)
    y_train = train["label"]
    y_test = test["label"]

    return x_train, x_test, y_train, y_test

# TODO Training and Validation!
# TODO make this more general!
# TODO I have to weight the labels! Assigning the most frequent label is not helpful!
# TODO: Ellbogen Methode um herauszufinden ob andere Cluster Zahlen eventuell mehr Sinn machen
# TODO: hyperparameters?
def kmeans(x_train, y_train):
    """ Semisupervised kmeans algorithm. Assigns most frequent snow label to cluster.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
    Returns:
        float: balanced_accuracy_score of training (for the moment)
    """
    # K-MEANS CLUSTERING
    cluster_num = len(LABELS)-3
    km = KMeans(n_clusters=cluster_num, init="random", n_init=cluster_num, random_state=42)
    # is fit_predict correct here?
    cluster_labels = km.fit_predict(x_train)

    # assign labels to clusters via y_train
    y_train_pred = cluster_labels
    for i in range(cluster_num):
        # mask where the cluster_labels are i
        mask = cluster_labels == i
        snow_label_i = np.argmax(np.bincount(y_train[mask]))
        y_train_pred[mask] = snow_label_i

    # training metrics
    return balanced_accuracy_score(y_true = y_train, y_pred=y_train_pred)


# TODO: a lot. optimize!!! a lot!
def gaussian_mix(x_train, y_train):
    """ Semisupervised Gaussian Mixture Algorithm. Assigns most frequent snow label to gaussians.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
    Returns:
        float: balanced_accuracy_score of training (for the moment)
    """
    # mixture model clustering
    n_gaussians = 10
    gm = GaussianMixture(n_components=n_gaussians, init_params="random", covariance_type="tied", random_state=42)
    gm_pred = gm.fit_predict(x_train)
    print(np.bincount(gm_pred))

    # assign labels to clusters via y_train
    y_train_pred = gm_pred
    for i in range(n_gaussians):
        # mask where the cluster_labels are i
        mask = gm_pred == i
        snow_label_i = np.argmax(np.bincount(y_train[mask]))
        y_train_pred[mask] = snow_label_i

    return balanced_accuracy_score(y_true = y_train, y_pred=y_train_pred)


def random_forest(x_train, y_train):
    """ Random Forest.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
    Returns:
        float: balanced_accuracy_score of training (for the moment)
    """
    rf = RandomForestClassifier(n_estimators=100,
                                criterion = "entropy",
                                bootstrap = True,
                                max_samples = 0.6,     # 60 % of the training data (None: all)
                                max_features = "sqrt", # uses sqrt(num_features) features
                                random_state = 42)
    rf_pred = rf.fit(x_train, y_train).predict(x_train)
    return balanced_accuracy_score(y_true = y_train, y_pred=rf_pred)

def svm(x_train, y_train, gamma="auto"):
    """ Support Vector Machine with Radial Basis functions as kernel.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        gamma (num or Str): gamma value for svm
    Returns:
        float: balanced_accuracy_score of training (for the moment)
    """
    svm = SVC(decision_function_shape='ovr', kernel="rbf", gamma=gamma, random_state=24)
    svm_pred = svm.fit(x_train, y_train).predict(x_train)
    print(np.unique(svm_pred, return_counts=True))
    print((svm_pred == y_train).sum())
    print(multilabel_confusion_matrix(y_train, svm_pred))
    return balanced_accuracy_score(y_true = y_train, y_pred=svm_pred)

def knn(x_train, y_train, n_neighbors):
    """ Support Vector Machine with Radial Basis functions as kernel.
    Parameters:
        x_train: Input data for training
        y_train: Target data for training
        n_neighbors: Number of neighbors to consider
    Returns:
        float: balanced_accuracy_score of training (for the moment)
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn_pred = knn.fit(x_train, y_train).predict(x_train)
    return balanced_accuracy_score(y_true = y_train, y_pred=knn_pred)

def main():
    # load dataframe with smp data
    smp = load_data("smp_lambda_delta_gradient.npz")
    #visualize_original_data(smp)
    x_train, x_test, y_train, y_test = my_train_test_split(smp)

    # prepare data for cross Validation
    # TODO: update for anns -> time series must stay intact
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

    # what I am ignoring at the moment:
    # TODO: balancing the dataset (...oversampling? VAE? What is the best to do here?)
    # TODO: correct cross-validation (needs to be done manual for ANNs)
    # TODO: Preprocessing: Standardization? Normalization?
    # TODO: correct metrics
    # TODO: visualization

    x_train = x_train.drop(["smp_idx"], axis=1)
    x_test = x_test.drop(["smp_idx"], axis=1)
    print(np.unique(y_train, return_counts=True))

    # kmeans clustering (does not work)
    kmeans_acc = kmeans(x_train, y_train, cv)

    # mixture model clustering (does not work)
    gm_acc = gaussian_mix(x_train, y_train, cv)

    # random forests (works)
    rf_acc = random_forest(x_train, y_train, cv)

    # works with very high gamma (overfitting) -> "auto" yields 0.75, still good and no overfitting
    svm_acc = svm(x_train, y_train, cv, gamma=5)

    # knn (works with weights=distance)
    knn_acc = knn(x_train, y_train, cv, n_neighbors=20)

    print(tabulate([["Kmeans", kmeans_acc], ["Gaussian Mixture", gm_acc], ["Random Forest", rf_acc], ["Support Vector Machine", svm_acc], ["K Nearest Neighbors", knn_acc]],
                    header=["Model", "Training Accuracy"], tablefmt="orgtbl"))

    # ONLY FOR CURIOUSITY (will be deleted)
    # linear support vector machines -> does not work but makes sense
    svl = LinearSVC(multi_class="ovr", C=0.99, random_state=42)
    svl_pred = svl.fit(x_train, y_train).predict(x_train)
    # print(np.unique(svl_pred, return_counts=True))
    # print((svl_pred == y_train).sum())
    print("Linear SVM Training Accuracy", balanced_accuracy_score(y_true = y_train, y_pred=svl_pred))

    # # naive bayes classifier -> does not work but makes sense
    gnb = GaussianNB()
    gnb_pred = gnb.fit(x_train, y_train).predict(x_train)
    print("Naive Bayes Classifier", balanced_accuracy_score(y_true = y_train, y_pred=gnb_pred))



if __name__ == "__main__":
    main()
