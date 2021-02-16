# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_parameters import LABELS
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
    x_train = train.drop(["label"], axis=1)
    x_test = test.drop(["label"], axis=1)
    y_train = train["label"]
    y_test = test["label"]
    print("Labels in training data:\n", y_train.value_counts())
    print("Labels in testing data:\n", y_test.value_counts())
    return x_train, x_test, y_train, y_test

def sum_up_labels(smp, labels, name, label_idx):
    """ Sums up the datapoints belonging to one of the classes in labels to one class.
    Parameters:
        smp (pd.DataFrame): a dataframe with the smp profiles
        labels (list): a list of Strings with the labels which should be united
        name (String): name of the new unified class
        label_idx (int): the number identifying which label it is
    Returns:
        pd.DataFrame: the updated smp
    """
    int_labels = [LABELS[label] for label in labels]
    smp.loc[smp["label"].isin(int_labels), "label"] = label_idx
    # add a new key value pair to labels (and antilabels and colors?)
    # TODO update: add this also to antilabels and colors
    LABELS[name] = label_idx
    return smp


def self_training(unlabelled_data, x_train, y_train, cv):
    """ Self training
    """
    return "blubb"

def label_spreading(unlabelled_data, x_train, y_train, cv):
    """ Label spreading
    """
    return "blubb"

# TODO put this in an own function
# TODO one parameter should be the table format of the output
def main():
    # 1. Load dataframe with smp data
    smp = load_data("smp_lambda_delta_gradient.npz")

    # prepare dataset of unlabelled data
    # TODO: fix this: CURRENTLY crushes for smp_lambda_delta_gradient
    unlabelled_smp = smp[(smp["label"] == 0)]
    # set unlabelled_smp label to -1
    unlabelled_smp.loc["label"] = -1
    unlabelled_smp_x = unlabelled_smp.drop(["label", "smp_idx"], axis=1)
    unlabelled_smp_x = unlabelled_smp_x.dropna()
    unlabelled_smp_y = pd.Series(np.repeat(-1, len(unlabelled_smp_x)))

    # TODO: maybe visualize some things only after normalization and standardization?
    # 2. Visualize the original data
    visualize_original_data(smp)
    exit(0)
    # 3. Sum up certain classes if necessary (alternative: do something to balance the dataset)
    smp = sum_up_labels(smp, ["df", "drift_end", "ifwp", "if", "sh"], name="rare", label_idx=18)

    # 4. Split up the data into training and test data
    x_train, x_test, y_train, y_test = my_train_test_split(smp)
    # reset internal panda index
    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # 5. Normalize and standardize the data
    # TODO

    # 6. Make crossvalidation split
    k = 3
    # Note: if we want to use StratifiedKFold, we can just hand over an integer to the functions
    cv_stratified = StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x_train, y_train)
    cv_stratified = list(cv_stratified)
    # yields a list of tuples with training and test indices
    cv = cv_manual(x_train, k) # in progress

    x_train = x_train.drop(["smp_idx"], axis=1)
    x_test = x_test.drop(["smp_idx"], axis=1)
    print(np.unique(y_train, return_counts=True))

    # 7. Call the models
    all_scores = []

    # A Baseline - majority class predicition
    print("Starting Baseline Model...")
    baseline_scores = majority_class_baseline(x_train, y_train, cv_stratified)
    all_scores.append(mean_kfolds(baseline_scores))
    print("...finished Baseline Model.")

    # B kmeans clustering (does not work)
    print("Starting K-Means Model...")
    kmeans_scores = kmeans(unlabelled_smp_x, x_train, y_train, cv_stratified, num_clusters=10, find_num_clusters="both", plot=True)
    all_scores.append(mean_kfolds(kmeans_scores))
    print("...finished K-Means Model.")
    # print(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))

    # C mixture model clustering ("diag" works best at the moment)
    print("Starting Gaussian Mixture Model...")
    gm_acc_diag = gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag", plot=True)
    all_scores.append(mean_kfolds(gm_acc_diag))
    print("...finished Gaussian Mixture Model.")

    print("Starting Baysian Gaussian Mixture Model...")
    bgm_acc_diag = bayesian_gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag")
    all_scores.append(mean_kfolds(bgm_acc_diag))
    print("...finished Bayesian Gaussian Mixture Model.")

    print(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql')) # latex_raw works as well
    exit(0)

    # # ARE TAKING TOO MUCH TIME
    # # D + E -> different data preparation necessary
    # # include unlabelled data points in x_train and y_train
    # x_train_all = pd.concat([x_train, unlabelled_smp_x])
    # y_train_all = pd.concat([y_train, unlabelled_smp_y])
    #
    # # D label spreading model
    # ls_model = LabelSpreading(kernel="knn", alpha=0.2, n_jobs=-1).fit(x_train_all, y_train_all)
    # y_pred = ls_model.predict(x_train_all)
    # ls_bal_acc = balanced_accuracy_score(y_pred, y_train_all)
    # print("Label Spreading Model, Training Accuracy: ", ls_bal_acc)
    # exit(0)
    #
    # # TODO it makes sense to use the best hyperparameter tuned models here!
    # # E self training model
    # svm = SVC(probability=True, gamma="auto")
    # st_model = SelfTrainingClassifier(svm, verbose=True).fit(x_train_all, y_train_all)
    # print("Hello3")
    # y_pred = st_model.predict(x_train_all)
    # st_bal_acc = balanced_accuracy_score(y_pred, y_train_all)
    # print("Self Training Classifier, Training Accuracy: ", st_bal_acc)
    #
    #
    # print(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))


    # F random forests (works)
    print("Starting Random Forest Model ...")
    rf_scores = random_forest(x_train, y_train, cv_stratified)
    all_scores.append(mean_kfolds(rf_scores))
    print("...finished Random Forest Model.")

    # G Support Vector Machines
    # works with very high gamma (overfitting) -> "auto" yields 0.75, still good and no overfitting
    print("Starting Support Vector Machine...")
    svm_scores = svm(x_train, y_train, cv, gamma="auto")
    all_scores.append(mean_kfolds(svm_scores))
    print("...finished Support Vector Machine.")

    # H knn (works with weights=distance)
    print("Starting K-Nearest Neighbours Model...")
    knn_scores = knn(x_train, y_train, cv, n_neighbors=20)
    all_scores.append(mean_kfolds(knn_scores))
    print("...finished K-Nearest Neighbours Model.")

    # I adaboost
    print("Starting AdaBoost Model...")
    ada_scores = ada_boost(x_train, y_train, cv)
    all_scores.append(mean_kfolds(ada_scores))
    print("...finished AdaBoost Model.")

    # J LSTM

    # K Encoder-Decoder

    # print the validation results
    # TODO maybe rename the table columns (validation results)
    print(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))
    with open('plots/tables/models_with_baseline.txt', 'w') as f:
        f.write(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))

    # 8. Visualize the results


if __name__ == "__main__":
    main()
