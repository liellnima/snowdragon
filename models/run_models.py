# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_parameters import LABELS, ANTI_LABELS, COLORS
from models.visualization import visualize_original_data, visualize_normalized_data # TODO or something like this
from models.cv_handler import cv_manual, mean_kfolds
from models.supervised_models import svm, random_forest, ada_boost, knn
from models.semisupervised_models import kmeans, gaussian_mix, bayesian_gaussian_mix, label_spreading, self_training
from models.baseline import majority_class_baseline
from models.supervised_models import testing
from models.helper_funcs import normalize, save_results, load_results
from models.anns import ann

import pickle
import random
import numpy as np
import pandas as pd
from tabulate import tabulate

# Other metrics: https://stats.stackexchange.com/questions/390725/suitable-performance-metric-for-an-unbalanced-multi-class-classification-problem
from sklearn.model_selection import train_test_split, StratifiedKFold #, cross_validate, cross_val_score, cross_val_predict


# TODO remove all the following imports!!!
from sklearn.neighbors import KNeighborsClassifier
from models.metrics import balanced_accuracy
from models.cv_handler import calculate_metrics_cv
from data_handling.data_preprocessing import remove_negatives
# from sklearn.multioutput import MultiOutputClassifier

# TODO plot confusion matrix beautifully (multilabel_confusion_matrix)
# TODO plot ROC AUC curve beautifully (roc_curve(y_true, y_pred))

def my_train_test_split(smp, test_size=0.2, train_size=0.8, return_smp_idx=True):
    """ Splits data into training and testing data
    Parameters:
        smp (df.DataFrame): Preprocessed smp data
        test_size (float): between 0 and 1, size of testing data
        train_size (float): between 0 and 1, size of training data
        return_smp_idx (bool): indicates whether smp_idx should be returned as well
    Returns:
        quadruple or hexuple of pd.DataFrames/Series: x_train, x_test, y_train, y_test, (smp_idx_train), (smp_idx_test)
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
    # drop the label and smp_idx
    x_train = train.drop(["label", "smp_idx"], axis=1)
    x_test = test.drop(["label", "smp_idx"], axis=1)
    y_train = train["label"]
    y_test = test["label"]
    smp_idx_train = train["smp_idx"]
    smp_idx_test = test["smp_idx"]
    print("Labels in training data:\n", y_train.value_counts())
    print("Labels in testing data:\n", y_test.value_counts())

    # reset internal panda index
    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    smp_idx_train.reset_index(drop=True, inplace=True)
    smp_idx_test.reset_index(drop=True, inplace=True)

    if return_smp_idx:
        return x_train, x_test, y_train, y_test, smp_idx_train, smp_idx_test
    else:
        return x_train, x_test, y_train, y_test

def sum_up_labels(smp, labels, name, label_idx, color="blue"):
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


def remove_nans_mosaic(smp):
    """ Very specific function to remove special nans and negative values. The selection was done manually.
    Don't use this function for a different dataset or for any kind of updated dataset.
    Params:
        smp (pd.DataFrame): SMP Dataframe
    Returns:
        pd.DataFrame: the same smp dataframe but with removed nans and without negative values
    """
    # remove profile 4002260.0 -> only nans and always the same negative mean value
    smp = smp.loc[smp["smp_idx"] != 4002260.0, :]
    # remove part of the profile 4002627, since the last half has only 0 as values and later negative values (and other nonsense)
    smp = smp.drop(range(625705, 625848), axis=0).reset_index(drop=True)
    # the rest can be solved by forward filling (checked this manually)
    smp = smp.fillna(method='ffill')
    # drop another row that contains a negative value for no reason
    smp = smp.drop(397667, axis=0).reset_index(drop=True)

    return smp

def normalize_mosaic(smp):
    """ Normalizes all the features that should be normalized in the data. Don't use this method for other data!
    Parameters:
        smp (pd.DataFrame): Mosaic SMP dataframe
    Returns:
        pd.DataFrame: the normalized version of the given dataframe
    """
    # Unchanged features: distance, pos_rel, dist_ground, smp_idx, label
    # Physical normalization: all force features between 0 and 45
    physical_features = [feature for feature in smp.columns if ("force" in feature) and (not "var" in feature)]
    smp = normalize(smp, physical_features, min=0, max=45)
    # Non physical normalization: (natural min 0) lambda: ; delta: ; L -> they are normalized on the complete dataset (too different results otherwise)
    non_physical_features = [feature for feature in smp.columns if ("lambda" in feature) or ("delta" in feature) or ("L" in feature)]
    for feature in non_physical_features:
        smp = normalize(smp, feature, min=0, max=smp[feature].max())
    # gradient
    smp = normalize(smp, "gradient", min=smp["gradient"].min(), max=smp["gradient"].max())
    # var features
    var_features = [feature for feature in smp.columns if ("var" in feature)]
    for feature in var_features:
        smp = normalize(smp, feature, min=0, max=smp[feature].max())
    # distances
    smp = normalize(smp, "distance", min=smp["distance"].min(), max=smp["distance"].max()) # max is 1187
    smp = normalize(smp, "dist_ground", min=smp["dist_ground"].min(), max=smp["dist_ground"].max())
    return smp

def preprocess_dataset(smp_file_name, visualize=False, sample_size_unlabelled=1000):
    """ Preprocesses the complete smp data and returns what is needed for the models.
    Parameters:
        smp_file_name (str): where the complete smp data is saved
        visualize (bool): if the data should be visualized before and after normalization
        sample_size_unlabelled (int): how many unlabelled samples should be included in x_train_all and y_train_all

    Returns:
        (dict): "x_train", "y_train", " x_test" and "y_test" are the prepared and normalized training and test data.
                "x_train_all" and "y_train_all" are both labelled and unlabelled data in one data set.
                "unlabelled_smp_x" is the complete unlabelled input data without any labelled data points.
                "cv", "cv_semisupervised" and "cv_timeseries" are cv splits for the supervised, semisupervised and ann models.
                "smp_idx_train" and "smp_idx_test" are the smp indiced corresponding to training and test data (correct order)
    """
    # 1. Load dataframe with smp data
    smp_org = load_data(smp_file_name)
    # remove nans
    smp_org = remove_nans_mosaic(smp_org)

    # 2. Visualize before normalization
    if visualize: visualize_original_data(smp_org)

    # 3. Normalize
    smp = normalize_mosaic(smp_org)

    # 4. Sum up certain classes if necessary (alternative: do something to balance the dataset)
    # (keep: 6, 3, 4, 12, 5, 16: rgwp, dh, dhid, dhwp, mfdh, pp)
    smp = sum_up_labels(smp, ["df", "ifwp", "if", "sh", "snow-ice", "mfcl", "mfsl", "mfcr"], name="rare", label_idx=17)

    # 5. Visualize the data after normalization
    if visualize: visualize_normalized_data(smp)

    # 6. Prepare unlabelled data for two of the semisupervised modles:
    # prepare dataset of unlabelled data
    unlabelled_smp = smp.loc[(smp["label"] == 0)].copy()
    # set unlabelled_smp label to -1
    unlabelled_smp.loc[:, "label"] = -1
    unlabelled_smp_x = unlabelled_smp.drop(["label", "smp_idx"], axis=1)
    unlabelled_smp_y = unlabelled_smp["label"]
    # sample in order to make it time-wise possible
    # OBSERVATION: the more data we include the worse the scores for the models become
    unlabelled_x = unlabelled_smp_x.sample(sample_size_unlabelled) # complete data: 650 326
    unlabelled_y = unlabelled_smp_y.sample(sample_size_unlabelled) # we can do this, because labels are only -1 anyway

    # 7. Split up the labelled data into training and test data
    x_train, x_test, y_train, y_test, smp_idx_train, smp_idx_test = my_train_test_split(smp)

    # For two of the semisupervised models: include unlabelled data points in x_train and y_train (test data stays the same!)
    x_train_all = pd.concat([x_train, unlabelled_x])
    y_train_all = pd.concat([y_train, unlabelled_y])

    # 8. Make crossvalidation split
    k = 3
    # Note: if we want to use StratifiedKFold, we can just hand over an integer to the functions
    cv_stratified = StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x_train, y_train)
    cv_stratified = list(cv_stratified)
    # Attention the cv fold for these two semi-supervised models is different from the other cv folds!
    cv_semisupervised = StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x_train_all, y_train_all)
    cv_semisupervised = list(cv_semisupervised)
    data = x_train.copy()
    target = y_train.copy()
    data["smp_idx"] = smp_idx_train
    cv_timeseries = cv_manual(data, target, k)
    #print(np.unique(y_train, return_counts=True))

    # what is needed for the models:
    # save this in a dictionary and save the dictionary as npz file
    prepared_data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
                     "x_train_all": x_train_all, "y_train_all": y_train_all, "unlabelled_smp_x": unlabelled_smp_x,
                     "cv": cv_stratified, "cv_semisupervised": cv_semisupervised, "cv_timeseries": cv_timeseries,
                     "smp_idx_train": smp_idx_train, "smp_idx_test": smp_idx_test}

    with open("preprocessed_data_dict.txt", "wb") as myFile:
        pickle.dump(prepared_data, myFile)

    return prepared_data

# TODO put this in tuning
# TODO make smp_all_03.npz a global variable
def run_single_model(model_type, data, **params):
    """ Runs a single model. Returns the results for this run.
    Parameters:
        model_type (str): Can be one of the following:
            "baseline", "kmeans", "gmm", "bmm", "label_spreading", "self_trainer",
             "rf", "svm", "knn", "easy_ensemble", "lstm", "blstm", "enc_dec"
        data (dict): dictionary produced by preprocess_dataset containing all necessary information
        **params: contains all necessary parameters for the wished model
    """

    print("Starting Model {}...".format(model_type))

    # # different cases for different models
    # if model_type == "baseline":
    #     scores = majority_class_baseline(**data, **params)
    #
    # elif model_type == "kmeans":
    #     # params: num_clusters (5), find_num_clusters ("sil", "acc", "both")
    #     scores = kmeans(cv=data["cv_semisupervised"], **data, **params)
    #
    # elif model_type == "gmm":
    #     # params: num_components (15), find_num_clusters ("bic", "acc", "both"), cov_type ("tied", "diag", "spherical", "full")
    #     # fixed: max_iter = 150, init_params="random" (could be also "k-means")
    #     scores = gaussian_mix(cv=data["cv_semisupervised"], **data, **params)
    #
    # elif model_type == "bmm":
    #     # params: num_components (15), cov_type ("tied", "diag", "spherical", "full")
    #     # fixed: max_iter = 150, init_params="random" (could be also "k-means")
    #     scores = bayesian_gaussian_mix(cv=data["cv_semisupervised"], **data, **params)
    #
    # elif model_type == "label_spreading":
    #     # params: kernel (knn or rbf), alpha (0.2)
    #     # fixed: max_iter = 100 (quite high compared to 30)
    #     scores = label_spreading(cv=data["cv_semisupervised"], **data, **params)
    #
    # elif model_type == "self_trainer":
    #     # params:
    #     scores = self_training(cv=data["cv_semisupervised"], **data, **params)
    #
    # elif model_type == "rf":
    #
    # elif model_type == "svm":
    #
    # elif model_type == "knn":
    #
    # elif model_type == "easy_ensemble":
    #
    # elif model_type == "lstm":
    #
    # elif model_type == "blstm":
    #
    # elif model_type == "enc_dec":
    #
    # else:
    #     print("No such model exists. Please choose one of the following models:\n")
    #     print(""" baseline, kmeans, gmm, bmm, label_spreading, self_trainer,
    #           rf, svm, knn, easy_ensemble, lstm, blstm, enc_dec""")

    scores = mean_kfolds(scores)
    # naming the parameters employed
    print(scores)
    print("...finished Model {}.\n".format(model_type))

    # just return the scores - saving is the job of another function
    return scores

def run_all_models(data, intermediate_file):
    """ All models are run here - the parameters are set manually within the function.
    Parameters:
        data (dict): dictionary produced by preprocess_dataset containing all necessary information
        intermediate_file (str): file_name for saving intermediate results
    """
    # unpack all necessary values from the preprocessing dictionary
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    x_train_all = data["x_train_all"]
    y_train_all = data["y_train_all"]
    unlabelled_smp_x = data["unlabelled_smp_x"]
    cv_stratified = data["cv"]
    cv_semisupervised = data["cv_semisupervised"]
    cv_timeseries = data["cv_timeseries"]
    smp_idx_train = data["smp_idx_train"]
    smp_idx_test = data["smp_idx_test"]

    # 9. Call the models
    all_scores = []

    # # A Baseline - majority class predicition
    # print("Starting Baseline Model...")
    # baseline_scores = majority_class_baseline(x_train, y_train, cv_stratified)
    # all_scores.append(mean_kfolds(baseline_scores))
    # save_results(intermediate_file, all_scores)
    # print("...finished Baseline Model.\n")
    #
    # # B kmeans clustering (does not work well)
    # # BEST cluster selection criterion: no difference, you can use either acc or sil (use sil in this case!)
    # print("Starting K-Means Model...")
    # kmeans_scores = kmeans(unlabelled_smp_x, x_train, y_train, cv_stratified, num_clusters=10, find_num_clusters="acc", plot=False)
    # all_scores.append(mean_kfolds(kmeans_scores))
    # save_results(intermediate_file, all_scores)
    # print("...finished K-Means Model.\n")
    #
    # # C mixture model clustering ("diag" works best at the moment)
    # # BEST cluster selection criterion: bic is slightly better than acc (generalization)
    # print("Starting Gaussian Mixture Model...")
    # gm_acc_diag = gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag", find_num_components="acc", plot=False)
    # all_scores.append(mean_kfolds(gm_acc_diag))
    # save_results(intermediate_file, all_scores)
    # print("...finished Gaussian Mixture Model.\n")
    #
    # print("Starting Baysian Gaussian Mixture Model...")
    # bgm_acc_diag = bayesian_gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag")
    # all_scores.append(mean_kfolds(bgm_acc_diag))
    # save_results(intermediate_file, all_scores)
    # print("...finished Bayesian Gaussian Mixture Model.\n")

    # TAKES A LOT OF TIME FOR COMPLETE DATA SET
    # D + E -> different data preparation necessary

    # # D label spreading model
    # print("Starting Label Spreading Model...")
    # ls_scores = label_spreading(x_train=x_train_all, y_train=y_train_all, cv=cv_semisupervised, name="LabelSpreading_1000")
    # all_scores.append(mean_kfolds(ls_scores))
    # save_results(intermediate_file, all_scores)
    # print("...finished Label Spreading Model.\n")
    #
    # # TODO it makes sense to use the best hyperparameter tuned models here!
    # # E self training model
    # print("Starting Self Training Classifier...")
    #svm = SVC(probability=True, gamma="auto")
    # knn = KNeighborsClassifier(n_neighbors = 20, weights = "distance")
    # st_scores = self_training(x_train=x_train_all, y_train=y_train_all, cv=cv_semisupervised, base_model=knn, name="SelfTraining_1000")
    # all_scores.append(mean_kfolds(st_scores))
    # print("...finished Self Training Classifier.\n")

    # # F random forests (works)
    # print("Starting Random Forest Model ...")
    # rf, rf_scores = random_forest(x_train, y_train, cv_stratified, visualize=False)
    # all_scores.append(mean_kfolds(rf_scores))
    # save_results(intermediate_file, all_scores)
    # #rf = random_forest(x_train, y_train, cv_stratified, visualize=False, only_model=True)
    # #rf_test_scores = testing(rf, x_train, y_train, x_test, y_test, smp_idx_train, smp_idx_test)
    # print("...finished Random Forest Model.\n")
    # #
    # # G Support Vector Machines
    # # works with very high gamma (overfitting) -> "auto" yields 0.75, still good and no overfitting
    # print("Starting Support Vector Machine...")
    # svm_scores = svm(x_train, y_train, cv_stratified, gamma="auto")
    # all_scores.append(mean_kfolds(svm_scores))
    # save_results(intermediate_file, all_scores)
    # print("...finished Support Vector Machine.\n")
    #
    # H knn (works with weights=distance)
    print("Starting K-Nearest Neighbours Model...")
    knn_scores = knn(x_train, y_train, cv_stratified, n_neighbors=20)
    all_scores.append(mean_kfolds(knn_scores))
    save_results(intermediate_file, all_scores)
    print("...finished K-Nearest Neighbours Model.\n")
    #
    # I adaboost
    print("Starting AdaBoost Model...")
    ada_scores = ada_boost(x_train, y_train, cv_stratified)
    all_scores.append(mean_kfolds(ada_scores))
    save_results(intermediate_file, all_scores)
    print("...finished AdaBoost Model.\n")

    # J LSTM
    print("Starting LSTM Model...")
    lstm_scores = ann(x_train, y_train, smp_idx_train, ann_type="lstm", cv=cv_timeseries, name="LSTM",
                      batch_size=32, epochs=10, rnn_size=25, dense_units=25, dropout=0.2, learning_rate=0.01)
    print(lstm_scores)
    all_scores.append(mean_kfolds(lstm_scores))
    save_results(intermediate_file, all_scores)
    print("...finished LSTM Model.\n")

    # K BLSTM
    print("Starting BLSTM Model...")
    #  cv can be a float, or a cv split
    blstm_scores = ann(x_train, y_train, smp_idx_train, ann_type="blstm", cv=cv_timeseries, name="BLSTM",
                       batch_size=32, epochs=10, rnn_size=25, dense_units=25, dropout=0.2, learning_rate=0.01)
    print(blstm_scores)
    all_scores.append(mean_kfolds(blstm_scores))
    save_results(intermediate_file, all_scores)
    print("...finished BLSTM Model.\n")

    # batch_size=6, epochs=50, learning_rate=0.01, plot_loss=plot_loss
    # ann_type=ann_type, rnn_size=100, dropout=0.2, dense_units=100

    # batch_size=32, epochs=10, learning_rate=0.01,
    # ann_type=ann_type, rnn_size=100, dropout=0, dense_units=0, plot_loss=plot_loss

    # J Encoder-Decoder
    print("Starting Encoder-Decoder Model...")
    #  cv can be a float, or a cv split
    encdec_scores = ann(x_train, y_train, smp_idx_train, ann_type="enc_dec", cv=cv_timeseries, name="ENC_DEC",
                       batch_size=32, epochs=10, rnn_size=25, dense_units=0, dropout=0.2, learning_rate=0.001,
                       attention=True, bidirectional=False)
    print(encdec_scores)
    all_scores.append(mean_kfolds(encdec_scores))
    save_results(intermediate_file, all_scores)
    print("...finished Encoder-Decoder Model.\n")

# parameters for this:
# data_dict (str): npz file name with dictionary or None, if no preprocessing file exists yet.
# TODO one parameter should be the table format of the output
def main():
    data_dict = "preprocessed_data_dict.txt"
    if data_dict is None:
        data = preprocess_dataset(smp_file_name="smp_all_03.npz")
    else:
        with open(data_dict, "rb") as myFile:
            data = pickle.load(myFile)

    #run_single_model(data)
    #exit(0)
    run_all_models(data, "all_results_test01_part03.txt")
    all_scores = load_results("all_results_test01.txt")

    # print the validation results
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

    with open('plots/tables/models_160smp_test01.txt', 'w') as f:
        f.write(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='psql'))

    with open('plots/tables/models_160smp_test01_latex.txt', 'w') as f:
        f.write(tabulate(pd.DataFrame(all_scores), headers='keys', tablefmt='latex_raw'))

    # Visualize the results
    # STILL MISSING

    #run_single_model(data_dict="preprocessed_data_dict.txt")




if __name__ == "__main__":
    random.seed(42)
    main()
