# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_parameters import LABELS, ANTI_LABELS, COLORS
from models.visualization import visualize_original_data # TODO or something like this
from models.cv_handler import cv_manual, mean_kfolds
from models.supervised_models import svm, random_forest, ada_boost, knn
from models.semisupervised_models import kmeans, gaussian_mix, bayesian_gaussian_mix
from models.baseline import majority_class_baseline

import numpy as np
import pandas as pd
from tabulate import tabulate

# Other metrics: https://stats.stackexchange.com/questions/390725/suitable-performance-metric-for-an-unbalanced-multi-class-classification-problem
from sklearn.model_selection import train_test_split, StratifiedKFold #, cross_validate, cross_val_score, cross_val_predict
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading

# TODO remove all the following imports!!!
from sklearn.neighbors import KNeighborsClassifier
from models.metrics import balanced_accuracy
from models.cv_handler import calculate_metrics_cv
# from sklearn.multioutput import MultiOutputClassifier

# TODO plot confusion matrix beautifully (multilabel_confusion_matrix)
# TODO plot ROC AUC curve beautifully (roc_curve(y_true, y_pred))

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
    # drop the label and also smp_idx (should not be used as an informative feature)
    x_train = train.drop(["label", "smp_idx"], axis=1)
    x_test = test.drop(["label", "smp_idx"], axis=1)
    y_train = train["label"]
    y_test = test["label"]
    print("Labels in training data:\n", y_train.value_counts())
    print("Labels in testing data:\n", y_test.value_counts())
    return x_train, x_test, y_train, y_test

def sum_up_labels(smp, labels, name, label_idx, color="orchid"):
    """ Sums up the datapoints belonging to one of the classes in labels to one class.
    Parameters:
        smp (pd.DataFrame): a dataframe with the smp profiles
        labels (list): a list of Strings with the labels which should be united
        name (String): name of the new unified class
        label_idx (int): the number identifying which label it is
        color (str): indicates which color the summed up class should get
    Returns:
        pd.DataFrame: the updated smp
    """
    int_labels = [LABELS[label] for label in labels]
    smp.loc[smp["label"].isin(int_labels), "label"] = label_idx
    # add a new key value pair to labels (and antilabels and colors?)
    # TODO are the labels really updated also in other files?
    LABELS[name] = label_idx
    ANTI_LABELS[label_idx] = name
    COLORS[label_idx] = color
    return smp

# TODO add parameters for base estimator etc.
def self_training(x_train, y_train, cv, name="SelfTrainingClassifier"):
    """ Self training - a semisupervised model.
    Parameters:
        x_train (pd.DataFrame): contains both the features of labelled and unlabelled data.
        y_train (pd.Series): contains the labels of the labelled and unlabelled data. Unlabelled data must have label -1.
        cv (list): List of training and testing tuples which contain the indiced for the different folds.
        name (str): Name/Description for the model.
    Returns:
        float: At the moment just the balanced accuracy
    """
    # TODO cv: use the same cv split but randomly assign the other unlabelled data pieces to the other cv folds
    #svm = SVC(probability=True, gamma="auto")
    knn = KNeighborsClassifier(n_neighbors = 20,
                               weights = "distance")
    st_model = SelfTrainingClassifier(knn, verbose=True).fit(x_train, y_train)
    # predict_proba possible
    #y_pred = st_model.predict(x_train)
    return calculate_metrics_cv(model=st_model, X=x_train, y_true=y_train, cv=cv, name=name)

# TODO add kernel and alpha parameter
def label_spreading(x_train, y_train, cv, name="LabelSpreading"):
    """ Label spreading - a semisupervised model.
    Parameters:
        x_train (pd.DataFrame): contains both the features of labelled and unlabelled data.
        y_train (pd.Series): contains the labels of the labelled and unlabelled data. Unlabelled data must have label -1.
        cv (list): List of training and testing tuples which contain the indiced for the different folds.
        name (str): Name/Description for the model.
    Returns:
        float: At the moment just the balanced accuracy
    """
    # TODO cv: use the same cv split but randomly assign the other unlabelled data pieces to the other cv folds
    ls_model = LabelSpreading(kernel="knn", alpha=0.2, n_jobs=-1).fit(x_train, y_train)
    #y_pred = ls_model.predict(x_train)
    return calculate_metrics_cv(model=ls_model, X=x_train, y_true=y_train, cv=cv, name=name)

# TODO put this in an own function
# TODO one parameter should be the table format of the output
def main():
    # 1. Load dataframe with smp data
    smp = load_data("smp_all_02.npz")#("smp_lambda_delta_gradient.npz")#
    # fill in nans (occur only for lambda_4, delta_4, lambda_12 and delta_12): use next occuring value
    smp = smp.fillna(method='ffill')

    # TODO: maybe visualize some things only after normalization and standardization?
    # 2. Visualize the original data
    #visualize_original_data(smp)

    # 3. Prepare data for two of the semisupervised modles:
    # prepare dataset of unlabelled data
    # TODO: fix this: CURRENTLY crushes for smp_lambda_delta_gradient
    unlabelled_smp = smp.loc[(smp["label"] == 0)].copy()
    # set unlabelled_smp label to -1
    unlabelled_smp.loc[:, "label"] = -1
    unlabelled_smp_x = unlabelled_smp.drop(["label", "smp_idx"], axis=1)
    unlabelled_smp_y = unlabelled_smp["label"]

    # sample in order to make it time-wise possible
    # OBSERVATION: the more data we include the worse the scores for the models become
    unlabelled_x = unlabelled_smp_x.sample(1000) # complete data: 650 326
    unlabelled_y = unlabelled_smp_y.sample(1000) # we can do this, because labels are only -1 anyway

    # 4. Sum up certain classes if necessary (alternative: do something to balance the dataset)
    # (keep: 6, 3, 4, 13, 5: rgwp, dh, dhid, mfcl, mfdh)
    smp = sum_up_labels(smp, ["df", "ifwp", "if", "sh", "snow-ice", "dhwp", "mfsl", "mfcr", "pp"], name="rare", label_idx=17)

    # 5. Split up the data into training and test data
    x_train, x_test, y_train, y_test = my_train_test_split(smp)

    # reset internal panda index
    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    # For two of the semisupervised models: include unlabelled data points in x_train and y_train (test data stays the same!)
    x_train_all = pd.concat([x_train, unlabelled_x])
    y_train_all = pd.concat([y_train, unlabelled_y])

    # 6. Normalize and standardize the data
    # TODO

    # 7. Make crossvalidation split
    k = 4
    # Note: if we want to use StratifiedKFold, we can just hand over an integer to the functions
    cv_stratified = StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x_train, y_train)
    cv_stratified = list(cv_stratified)
    # Attention the cv fold for these two semi-supervised models is different from the other cv folds!
    cv_semisupervised = StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x_train_all, y_train_all)
    cv_semisupervised = list(cv_semisupervised)
    #cv = cv_manual(x_train, k) # in progress
    print(np.unique(y_train, return_counts=True))

    # 8. Call the models
    all_scores = []

    # A Baseline - majority class predicition
    print("Starting Baseline Model...")
    baseline_scores = majority_class_baseline(x_train, y_train, cv_stratified)
    all_scores.append(mean_kfolds(baseline_scores))
    print("...finished Baseline Model.\n")

    # B kmeans clustering (does not work well)
    # BEST cluster selection criterion: no difference, you can use either acc or sil (use sil in this case!)
    print("Starting K-Means Model...")
    kmeans_scores = kmeans(unlabelled_smp_x, x_train, y_train, cv_stratified, num_clusters=10, find_num_clusters="acc", plot=False)
    all_scores.append(mean_kfolds(kmeans_scores))
    print("...finished K-Means Model.\n")

    # C mixture model clustering ("diag" works best at the moment)
    # BEST cluster selection criterion: bic is slightly better than acc (generalization)
    print("Starting Gaussian Mixture Model...")
    gm_acc_diag = gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag", find_num_components="acc", plot=False)
    all_scores.append(mean_kfolds(gm_acc_diag))
    print("...finished Gaussian Mixture Model.\n")

    print("Starting Baysian Gaussian Mixture Model...")
    bgm_acc_diag = bayesian_gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag")
    all_scores.append(mean_kfolds(bgm_acc_diag))
    print("...finished Bayesian Gaussian Mixture Model.\n")

    # TAKES A LOT OF TIME FOR COMPLETE DATA SET
    # D + E -> different data preparation necessary

    # D label spreading model
    print("Starting Label Spreading Model...")
    ls_scores = label_spreading(x_train=x_train_all, y_train=y_train_all, cv=cv_semisupervised, name="LabelSpreading_1000")
    all_scores.append(mean_kfolds(ls_scores))
    print("...finished Label Spreading Model.\n")

    # TODO it makes sense to use the best hyperparameter tuned models here!
    # E self training model
    print("Starting Self Training Classifier...")
    st_scores = self_training(x_train=x_train_all, y_train=y_train_all, cv=cv_semisupervised, name="SelfTraining_1000")
    all_scores.append(mean_kfolds(st_scores))
    print("...finished Self Training Classifier.\n")

    # F random forests (works)
    print("Starting Random Forest Model ...")
    rf_scores = random_forest(x_train, y_train, cv_stratified)
    all_scores.append(mean_kfolds(rf_scores))
    print("...finished Random Forest Model.\n")

    # G Support Vector Machines
    # works with very high gamma (overfitting) -> "auto" yields 0.75, still good and no overfitting
    print("Starting Support Vector Machine...")
    svm_scores = svm(x_train, y_train, cv_stratified, gamma="auto")
    all_scores.append(mean_kfolds(svm_scores))
    print("...finished Support Vector Machine.\n")

    # H knn (works with weights=distance)
    print("Starting K-Nearest Neighbours Model...")
    knn_scores = knn(x_train, y_train, cv_stratified, n_neighbors=20)
    all_scores.append(mean_kfolds(knn_scores))
    print("...finished K-Nearest Neighbours Model.\n")

    # I adaboost
    print("Starting AdaBoost Model...")
    ada_scores = ada_boost(x_train, y_train, cv_stratified)
    all_scores.append(mean_kfolds(ada_scores))
    print("...finished AdaBoost Model.\n")

    # J LSTM

    # K Encoder-Decoder

    # 9. print the validation results
    all_scores = pd.DataFrame(all_scores).rename(columns={"test_balanced_accuracy": "test_bal_acc",
                                                         "train_balanced_accuracy": "train_bal_acc",
                                                         "test_recall": "test_rec",
                                                         "train_recall": "train_rec",
                                                         "test_precision": "test_prec",
                                                         "train_precision": "train_prec",
                                                         "train_roc_auc": "train_roc",
                                                         "test_roc_auc": "test_roc",
                                                         "train_log_loss": "train_ll",
                                                         "test_log_loss": "test_ll"})
    print(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))
    with open('plots/tables/models_160smp.txt', 'w') as f:
        f.write(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))

    with open('plots/tables/models_160smp_latex.txt', 'w') as f:
        f.write(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='latex_raw'))

    # 10. Visualize the results


if __name__ == "__main__":
    main()
