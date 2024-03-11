# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_parameters import LABELS, ANTI_LABELS, COLORS
from models.cv_handler import cv_manual, mean_kfolds
from models.supervised_models import svm, random_forest, ada_boost, knn
from models.semisupervised_models import kmeans, gaussian_mix, bayesian_gaussian_mix, label_spreading, self_training
from models.baseline import majority_class_baseline
from models.helper_funcs import normalize, save_results, load_results, reverse_normalize, int_to_idx
from models.anns import ann, get_ann_model
from models.evaluation import testing, train_single_model
from visualization.plot_data import all_in_one_plot
from visualization.plot_profile import smp_labelled
from visualization.run_visualization import visualize_original_data, visualize_normalized_data # TODO or something like this
from tuning.tuning_parameters import BEST_PARAMS

import joblib
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow.keras.models

# surpress pandas warning SettingWithCopyWarning
pd.options.mode.chained_assignment = None

from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold #, cross_validate, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from data_handling.data_parameters import SMP_ORIGINAL_NPZ, SMP_NORMALIZED_NPZ, SMP_PREPROCESSED_TXT

# Explanation
# python -m models.run_models --preprocess

# parser
parser = argparse.ArgumentParser(description="Evaluate or validate the models performances. If not already done, preprocess the data beforehand.")

# File arguments
parser.add_argument("--smp_npz", default=SMP_ORIGINAL_NPZ, type=str, help="Name of the united npz file")
parser.add_argument("--preprocess_file", default=SMP_PREPROCESSED_TXT, type=str, help="Name of the txt file where the preprocessed data is stored.")
# what-is-done-arguments
parser.add_argument("--preprocess", action="store_true", help="Data must be preprocessed and stored in 'preprocessing_file'.")
parser.add_argument("--evaluate", action="store_true", help="Models are evaluated. Data from 'smp_npz' is used.")
parser.add_argument("--validate", action="store_true", help="Models are validated. Data from 'smp_npz' is used.")
parser.add_argument("--train_and_store", action="store_true", help="Models are trained and stored. Data from 'smp_npz' is used.")
parser.add_argument("--models", default=["all"], nargs='+', help="List of models that should be trained and stored.")

# TODO put most of those functions here in helper_funcs

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
    # Unchanged features: (distance), pos_rel, dist_ground, smp_idx, label
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

def preprocess_dataset(smp_file_name, output_file=None, visualize=False, sample_size_unlabelled=1000, tsne=None):
    """ Preprocesses the complete smp data and returns what is needed for the models.
    Parameters:
        smp_file_name (str): where the complete smp data is saved
        output_file (str): where the resulting dict should be saved. If None it
            is not saved.
        visualize (bool): if the data should be visualized before and after
            normalization
        sample_size_unlabelled (int): how many unlabelled samples should be
            included in x_train_all and y_train_all
        tsne (int): None means no dim reduction. int indicated that the data's
            dimensionality should be reduced with the help of tsne.
            The number indicates how many dimensions should remain.
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

    # CHANGE HERE - YOUR FAVORITE LABELS IN THE DATA!

    # 4. Sum up certain classes if necessary (alternative: do something to balance the dataset)
    # check: which profiles (and how many) have melted datapoints?
    #print("Profiles in which melted forms occur:")
    meltform_profiles = smp.loc[(smp["label"] == LABELS["mfcl"]) | (smp["label"] == LABELS["mfcr"]), "smp_idx"].unique()
    print(smp.loc[(smp["label"] == LABELS["sh"]), "smp_idx"].unique())
    #meltform_profiles_str = [int_to_idx(profile) for profile in meltform_profiles]
    # exclude these profiles!
    smp = smp[~smp["smp_idx"].isin(meltform_profiles)]
    # rename all df points to pp
    smp.loc[smp["label"] == LABELS["df"], "label"] = LABELS["pp"]
    # keep: 6, 3, 4, 12, 5, 16, 8, 10: rgwp, dh, dhid, dhwp, mfdh, pp(, if, sh)
    smp = sum_up_labels(smp, ["if", "sh"], name="rare", label_idx=17)

    print(smp["label"].value_counts())

    # save normalized data
    dict = smp.to_dict(orient="list")
    np.savez_compressed(SMP_NORMALIZED_NPZ, **dict)

    # 5. Visualize the data after normalization
    if visualize: visualize_normalized_data(smp)

    # if wished, make dimension reduction here!
    if tsne is not None:
        labels = smp["label"]
        indices = smp["smp_idx"]
        smp_x = smp.drop(["label", "smp_idx"], axis=1)
        tsne_model = TSNE(n_components=tsne, verbose=1, perplexity=40, n_iter=300, random_state=42)
        tsne_results = tsne_model.fit_transform(smp_x)
        smp_with_tsne = {"label": y, "smp_idx": idx}

        for i in range(tsne):
            smp_with_tsne["tsne" + str(i)] = tsne_results[:, i]

        smp = pd.DataFrame(smp_with_tsne)

    print(smp)
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
    k = 5
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
                     "x_train_all": x_train_all, "y_train_all": y_train_all, "unlabelled_data": unlabelled_smp_x,
                     "cv": cv_stratified, "cv_semisupervised": cv_semisupervised, "cv_timeseries": cv_timeseries,
                     "smp_idx_train": smp_idx_train, "smp_idx_test": smp_idx_test}

    if output_file is not None:
        with open(output_file, "wb") as myFile:
            pickle.dump(prepared_data, myFile)

    return prepared_data

def get_single_model(model_type, data, **params):
    """ Get a single model. Returns the model.
    Parameters:
        model_type (str): Can be one of the following:
            "baseline", "kmeans", "gmm", "bmm", "label_spreading", "self_trainer",
             "rf", "svm", "knn", "easy_ensemble", "lstm", "blstm", "enc_dec"
        data (dict): dictionary produced by preprocess_dataset containing all necessary information
        **params: contains all necessary parameters for the wished model
    Returns:
        model
    """
    # different cases for different models
    if model_type == "baseline":
        #model = majority_class_baseline(**data, **params)
        model = None

    elif model_type == "kmeans":
        model = kmeans(**data, **params, only_model=True)

    elif model_type == "gmm":
        model = gaussian_mix(**data, **params, only_model=True)

    elif model_type == "bmm":
        model = bayesian_gaussian_mix(**data, **params, only_model=True)

    elif model_type == "label_spreading":
        model = label_spreading(**data, **params, only_model=True)

    elif model_type == "self_trainer":
        if params["base_model"] is None:
            params["base_model"] = KNeighborsClassifier(n_neighbors = 10, weights = "distance")
        elif params["base_model"] == "best":
            params["base_model"] = BalancedRandomForestClassifier(
                                        n_estimators = 500,
                                        criterion = "entropy",
                                        bootstrap = True,
                                        max_samples = 0.6, # 60 % of the training data (None: all)
                                        max_features = "sqrt", # uses sqrt(num_features) features
                                        class_weight = "balanced", # balanced_subsample computes weights based on bootstrap sample
                                        random_state = 42) #random state might not work
        model = self_training(**data, **params, only_model=True)

    elif (model_type == "rf") or (model_type == "rf_bal"):
        model = random_forest(**data, **params, only_model=True)

    elif model_type == "svm":
        model = svm(**data, **params, only_model=True)

    elif model_type == "knn":
        model = knn(**data, **params, only_model=True)

    elif model_type == "easy_ensemble":
        model = ada_boost(**data, **params, only_model=True)

    elif model_type == "lstm":
        model = get_ann_model(**data, **params, ann_type="lstm")

    elif model_type == "blstm":
        model = get_ann_model(**data, **params, ann_type="blstm")

    elif model_type == "enc_dec":
        model = get_ann_model(**data, **params, ann_type="enc_dec")

    else:
        print("No such model exists. Please choose one of the following models:\n")
        print(""" baseline, kmeans, gmm, bmm, label_spreading, self_trainer,
              rf, svm, knn, easy_ensemble, lstm, blstm, enc_dec""")

    return model

def run_single_model(model_type, data, name=None, **params):
    """ Runs a single model. Returns the results for this run.
    Parameters:
        model_type (str): Can be one of the following:
            "baseline", "kmeans", "gmm", "bmm", "label_spreading", "self_trainer",
             "rf", "svm", "knn", "easy_ensemble", "lstm", "blstm", "enc_dec"
        data (dict): dictionary produced by preprocess_dataset containing all necessary information
        **params: contains all necessary parameters for the wished model
    """
    if name is None:
        name = model_type

    print("Starting Model {}...".format(model_type))

    # different cases for different models
    if model_type == "baseline":
        scores = majority_class_baseline(**data, **params, name=name)

    elif model_type == "kmeans":
        # params: num_clusters (5), find_num_clusters ("sil", "acc", "both")
        scores = kmeans(**data, **params, name=name)

    elif model_type == "gmm":
        # params: num_components (15), find_num_clusters ("bic", "acc", "both"), cov_type ("tied", "diag", "spherical", "full")
        # fixed: max_iter = 150, init_params="random" (could be also "k-means")
        scores = gaussian_mix(**data, **params, name=name)

    elif model_type == "bmm":
        # params: num_components (15), cov_type ("tied", "diag", "spherical", "full")
        # fixed: max_iter = 150, init_params="random" (could be also "k-means")
        scores = bayesian_gaussian_mix(**data, **params, name=name)

    elif model_type == "label_spreading":
        # params: kernel (knn or rbf), alpha (0.2)
        # fixed: max_iter = 100 (quite high compared to 30)
        scores = label_spreading(**data, **params, name=name)

    elif model_type == "self_trainer":
        if params["base_model"] is None:
            params["base_model"] = KNeighborsClassifier(n_neighbors = 10, weights = "distance")
        elif params["base_model"] == "best":
            params["base_model"] = BalancedRandomForestClassifier(
                                        n_estimators = 500,
                                        criterion = "entropy",
                                        bootstrap = True,
                                        max_samples = 0.6, # 60 % of the training data (None: all)
                                        max_features = "sqrt", # uses sqrt(num_features) features
                                        class_weight = "balanced", # balanced_subsample computes weights based on bootstrap sample
                                        random_state = 42) #random state might not work
        # params: criterion (threshold or k best -> set k or threshold), base_estimator (best model, but should be fast)
        # fixed: max_iter (None means until all unlabelled samples are labelled)
        scores = self_training(**data, **params, name=name)

    elif (model_type == "rf") or (model_type == "rf_bal"):
        # params: n_estimators, criterion (entropy, gini), max_features (sqrt, log2),
        #          max_samples (float, 0.6), resample (bool for balanced RF)
        # fixed: bootstrap=True, class_weight=balanced
        scores = random_forest(**data, **params, name=name)

    elif model_type == "svm":
        # params: decision_function_shape (ovr, ovo), gamma (scale, auto or float), kernel(linear, poly, rbf, sigmoid)
        #         kernel -> fix also degree for polynomial kernel, C?
        # fixed: class_weight=balanced, C=0.95
        scores = svm(**data, **params, name=name)

    elif model_type == "knn":
        # params: n_neighbors (int, number of neighbors),
        # fixed: weights (distance or uniform)
        scores = knn(**data, **params, name=name)

    elif model_type == "easy_ensemble":
        # params: n_estimators
        # fixed: sampling_strategy = not_majority or all -> find out
        scores = ada_boost(**data, **params, name=name)

    elif model_type == "lstm":
        # params: batch_size, epochs, learning_rate, rnn_size, dropout, dense_units
        # fixed: all the rest
        scores = ann(**data, **params, ann_type="lstm", name=name)

    elif model_type == "blstm":
        # params: batch_size, epochs, learning_rate, rnn_size, dropout, dense_units
        # fixed: all the rest
        scores = ann(**data, **params, ann_type="blstm", name=name)

    elif model_type == "enc_dec":
        # params: batch_size, epochs, learning_rate, rnn_size, dropout, dense_units,
        #         bidirectional, attention, regularizer
        # fixed: all the rest
        scores = ann(**data, **params, ann_type="enc_dec", name=name)

    else:
        print("No such model exists. Please choose one of the following models:\n")
        print(""" baseline, kmeans, gmm, bmm, label_spreading, self_trainer,
              rf, svm, knn, easy_ensemble, lstm, blstm, enc_dec""")

    scores = mean_kfolds(scores)

    if not "train_roc_auc" in scores: scores["train_roc_auc"] = float("nan")
    if not "test_roc_auc" in scores: scores["test_roc_auc"] = float("nan")
    if not "train_log_loss" in scores: scores["train_log_loss"] = float("nan")
    if not "test_log_loss" in scores: scores["test_log_loss"] = float("nan")

    print("...finished Model {}.\n".format(model_type))

    # just return the scores - saving is the job of another function
    # also job for others: naming the parameters employed and print scores
    return scores

def train_and_store_models(data, models=["all"], **kwargs):
    """ Trains and stores all the model in a given list with the params
    stored in BEST_PARAMS (see and change in tuning_parameters.py). Models are
    stored in the dir "/models/stored_models/".

    Parameters:
        data (dict): Created during data preprocessing and can be loaded e.g.
            from "data/preprocessed_data_k5.txt" (happens in run_models main func).
        models (list): list of models that should be trained and stored. Default
            is ["all"], i.e. all models are trained and stored
    """
    if models == ["all"]:
        all_models = ["baseline", "kmeans", "gmm", "bmm",
                      "rf", "rf_bal", "svm", "knn", "easy_ensemble",
                      "self_trainer", "label_spreading",
                      "lstm", "blstm", "enc_dec"]
    else:
        all_models = models

    type_implementation = {"rf": "scikit", "rf_bal": "scikit", "svm": "scikit", "knn": "scikit",
                           "easy_ensemble": "scikit", "self_trainer": "scikit",
                           "label_spreading": "scikit", "lstm": "keras",
                           "blstm": "keras", "enc_dec": "keras",
                           "baseline": "baseline", "kmeans": "semi_manual",
                           "gmm": "semi_manual", "bmm": "semi_manual"}

    for model_type in all_models:
        print("Training and Storing {} Model ...\n".format(model_type))
        model = get_single_model(model_type=model_type, data=data,
                                 **BEST_PARAMS[model_type])

        # fit the models
        if (model_type != "label_spreading") and (model_type != "self_trainer"):
            fitted_model = train_single_model(model, data["x_train"],
                                              data["y_train"], data["smp_idx_train"],
                                              impl_type=type_implementation[model_type],
                                              **BEST_PARAMS[model_type])

        else:
            fitted_model = train_single_model(model, data["x_train_all"],
                                              data["y_train_all"], data["smp_idx_train"],
                                              impl_type=type_implementation[model_type],
                                              **BEST_PARAMS[model_type])

        if type_implementation[model_type] != "keras":
            # store the models
            with open("models/stored_models/" + model_type + ".model", "wb") as handle:
                if model_type == "rf":
                    joblib.dump(fitted_model, handle, compress=9)
                else:
                    pickle.dump(fitted_model, handle)
        else:
            fitted_model.save("models/stored_models/" + model_type + ".hdf5")


def evaluate_all_models(data, file_scores=None, file_scores_lables=None, overwrite_tables=True, **params):
    """ Evaluating each model. Parameters for models are given in params.
    Results can be saved intermediately in a file.
    Parameters:
        data (dict): dictionary produced by preprocess_dataset containing all necessary information
        file_scores (path): where to save results intermediately
        file_scores_labels (path): where to save the labels of the results intermediately
        overwrite_tables (bool): If false the tables are not produced newly
        **params: A list for all necessary parameters for the models.
    """
    # set plotting variables:
    # no special labels order (default ascending) and name must be set individually
    save_overall_metrics = True
    smoothing = 0 # window for smoothing (later: one could also set different smoothing parameters for the different models!)
    plotting = {"annot": "eval", "roc_curve": True, "confusion_matrix": True,
                "one_plot": True, "pair_plots": True, "only_preds": True, "only_trues": False,
                "plot_list": None, "bog_plot_preds": "output/evaluation/", "bog_plot_trues": "output/evaluation/"}
    plotting = {"annot": "eval", "roc_curve": False, "confusion_matrix": False,
                "one_plot": True, "pair_plots": False, "only_preds": False, "only_trues": False,
                "plot_list": None, "bog_plot_preds": None, "bog_plot_trues": None}

    folders = {"rf": "output/evaluation/rf",
               "rf_bal": "output/evaluation/rf_bal",
               "svm": "output/evaluation/svm",
               "knn": "output/evaluation/knn",
               "easy_ensemble": "output/evaluation/easy_ensemble",
               "self_trainer": "output/evaluation/self_trainer",
               "label_spreading": "output/evaluation/label_spreading",
               "lstm": "output/evaluation/lstm",
               "blstm": "output/evaluation/blstm",
               "enc_dec": "output/evaluation/enc_dec",
               "baseline": "output/evaluation/baseline",
               "kmeans": "output/evaluation/kmeans",
               "gmm": "output/evaluation/gmm",
               "bmm": "output/evaluation/bmm"}
    # folders = {"knn": "output/evaluation/knnSmoothingTest"}

    type_implementation = {"rf": "scikit", "rf_bal": "scikit", "svm": "scikit", "knn": "scikit",
                           "easy_ensemble": "scikit", "self_trainer": "scikit",
                           "label_spreading": "scikit", "lstm": "keras",
                           "blstm": "keras", "enc_dec": "keras",
                           "baseline": "baseline", "kmeans": "semi_manual",
                           "gmm": "semi_manual", "bmm": "semi_manual"}


    all_models = ["baseline", "kmeans", "gmm", "bmm",
                  "rf", "rf_bal", "svm", "knn", "easy_ensemble",
                  "self_trainer", "label_spreading",
                  "lstm", "blstm", "enc_dec"]
    all_names = ["Majority Vote", "K-means", "Gaussian Mixture Model", "Bayesian Gaussian Mixture Model",
                 "Random Forest", "Balanced Random Forest", "Support Vector Machine", "K-nearest Neighbors", "Easy Ensemble",
                 "Self Trainer", "Label Propagation",
                 "LSTM", "BLSTM", "Encoder Decoder"]
    all_models = ["rf_bal", "lstm", "enc_dec", "self_trainer"]
    all_names = ["Balanced Random Forest", "LSTM", "Encoder Decoder", "Self Trainer"]
    # save bogplot for true predictions and all true smps in the folder above
    if (plotting["bog_plot_trues"] is not None) or (plotting["only_trues"]):
        # get important vars
        y_test = data["y_test"]
        smp_idx_test = data["smp_idx_test"]
        x_test = data["x_test"]
        # collect the smp true data
        smp_trues = []
        smp_names = smp_idx_test.unique()
        for smp_name in smp_names:
            smp = pd.DataFrame({"mean_force": x_test["mean_force"], "distance": x_test["distance"], "label": y_test, "smp_idx": smp_idx_test})
            smp = reverse_normalize(smp, "mean_force", min=0, max=45)
            smp = reverse_normalize(smp, "distance", min=0, max=1187)
            smp_wanted = smp[smp["smp_idx"] == smp_name]
            smp_trues.append(smp_wanted)
        all_smp_trues = pd.concat(smp_trues)
        print(all_smp_trues)

        # create the only_trues
        if plotting["only_trues"]:
            print("Plotting the observed SMP Profiles:")
            # create trues folder if it doesnt exist yet
            path_to_truth = Path.cwd() / "output/evaluation/trues"
            if not path_to_truth.is_dir():
                path_to_truth.mkdir(parents=True, exist_ok=True)
            for smp_name, smp_true in tqdm(zip(smp_names, smp_trues), total=len(smp_names)):
                smp_name_str = int_to_idx(smp_name)
                save_file = "output/evaluation/trues/smp_" + smp_name_str + ".png"
                smp_labelled(smp_true, smp_name, title="{} SMP Profile Observed\n".format(smp_name_str), file_name=save_file)
                # set the plotting value to False now
                plotting["only_trues"] = False

        # create the bogplot
        if plotting["bog_plot_trues"] is not None:
            print("Plotting the Bogplot of all observed SMP Profiles:")
            save_file = plotting["bog_plot_trues"] + "/bogplot_trues.png"
            all_in_one_plot(all_smp_trues, show_indices=False, sort=True,
                            title="All Observed SMP Profiles of the Testing Data", file_name=save_file)
            plotting["bog_plot_trues"] = None

    all_scores = []
    all_scores_per_label = []

    for model_type, name in zip(all_models, all_names):
        print("Evaluating {} Model ...\n".format(name))
        model = get_single_model(model_type=model_type, data=data,
                                 **BEST_PARAMS[model_type])
        # usual case
        if (model_type != "label_spreading") and (model_type != "self_trainer"):
            scores = testing(model, **data, name=name,
                             impl_type=type_implementation[model_type],
                             save_dir=folders[model_type],
                             save_visualization_data=True,
                             smoothing=smoothing, printing=True,
                             **plotting, **BEST_PARAMS[model_type])
        # if the semi supervised sciktit models are used, x_train and y_train
        # must be x_train_all and y_train_all
        else:
            scores = testing(model, data["x_train_all"], data["y_train_all"],
                             data["x_test"], data["y_test"],
                             data["smp_idx_train"], data["smp_idx_test"],
                             name=name, impl_type=type_implementation[model_type],
                             save_dir=folders[model_type],
                             save_visualization_data=True,
                             smoothing=smoothing, printing=True,
                             **plotting, **BEST_PARAMS[model_type])

        all_scores.append(scores[0])
        all_scores_per_label.append(scores[1])
        if file_scores is not None: save_results(file_scores, all_scores)
        if file_scores_lables is not None: save_results(file_scores_lables, all_scores_per_label)
        print("...finished {} Model.\n".format(name))

    if not overwrite_tables:
        print("Existing before overwriting tabular results")
        exit(0)

    # print all general scores and save them in evaluation
    print(tabulate(pd.concat(all_scores, axis=0, ignore_index=True), headers="keys", tablefmt="psql"))
    if save_overall_metrics:
        with open("output/scores/all_scores_psql.txt", 'w') as f:
            f.write(tabulate(pd.concat(all_scores, axis=0, ignore_index=True), headers="keys", tablefmt="psql"))
        with open("output/scores/all_scores_latex.txt", 'w') as f:
            f.write(tabulate(pd.concat(all_scores, axis=0, ignore_index=True), headers="keys", tablefmt="latex_raw"))
        pd.concat(all_scores, axis=0, ignore_index=True).to_csv("output/scores/all_scores.csv")

    # here, we can pick out the interesting stuff, like comparing the labels
    # based on accuracy  and precision for all models
    str_labels = [ANTI_LABELS[label] for label in [3, 4, 5, 6, 12, 16, 17]]
    acc_per_label = pd.DataFrame(columns=str_labels)
    prec_per_label = pd.DataFrame(columns=str_labels)
    for i, model in enumerate(all_scores_per_label):
        acc_per_label.loc[i] = model["eval_accuracy"]
        prec_per_label.loc[i] = model["eval_precision"]
    acc_per_label["model"] = [model_coll["model"] for model_coll in all_scores_per_label]
    prec_per_label["model"] = [model_coll["model"] for model_coll in all_scores_per_label]

    # save acc
    print("Accuracies:")
    print(tabulate(acc_per_label, headers="keys", showindex=False, tablefmt="psql"))
    if save_overall_metrics:
        with open("output/scores/acc_labels_psql.txt", 'w') as f:
            f.write(tabulate(acc_per_label, headers="keys", showindex=False, tablefmt="psql"))
        with open("output/scores/acc_labels_latex.txt", 'w') as f:
            f.write(tabulate(acc_per_label, headers="keys", showindex=False, tablefmt="latex_raw"))
        acc_per_label.to_csv("output/scores/acc_labels.csv")

    # save prec
    print("Precisions:")
    print(tabulate(prec_per_label, headers="keys", showindex=False, tablefmt="psql"))
    if save_overall_metrics:
        with open("output/scores/prec_labels_psql.txt", 'w') as f:
            f.write(tabulate(prec_per_label, headers="keys", showindex=False, tablefmt="psql"))
        with open("output/scores/prec_labels_latex.txt", 'w') as f:
            f.write(tabulate(prec_per_label, headers="keys", showindex=False, tablefmt="latex_raw"))
        prec_per_label.to_csv("output/scores/prec_labels.csv")




# TODO make a validation_all_models and evaluate_all_models function
# think about using run_single_model and creating a eval_single_model for this
def validate_all_models(data, intermediate_file=None):
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
    unlabelled_smp_x = data["unlabelled_data"]
    cv_stratified = data["cv"]
    cv_semisupervised = data["cv_semisupervised"]
    cv_timeseries = data["cv_timeseries"]
    smp_idx_train = data["smp_idx_train"]
    smp_idx_test = data["smp_idx_test"]

    # 9. Call the models
    all_scores = []

    # A Baseline - majority class predicition
    print("Starting Baseline Model...")
    baseline_scores = majority_class_baseline(x_train, y_train, cv_stratified)
    all_scores.append(mean_kfolds(baseline_scores))
    print(mean_kfolds(baseline_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished Baseline Model.\n")

    # B kmeans clustering (does not work well)
    # BEST cluster selection criterion: no difference, you can use either acc or sil (use sil in this case!)
    print("Starting K-Means Model...")
    kmeans_scores = kmeans(unlabelled_smp_x, x_train, y_train, cv_stratified, num_clusters=10, find_num_clusters="acc", plot=False)
    all_scores.append(mean_kfolds(kmeans_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished K-Means Model.\n")

    # C mixture model clustering ("diag" works best at the moment)
    # BEST cluster selection criterion: bic is slightly better than acc (generalization)
    print("Starting Gaussian Mixture Model...")
    gm_acc_diag = gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag", find_num_components="acc", plot=False)
    all_scores.append(mean_kfolds(gm_acc_diag))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished Gaussian Mixture Model.\n")

    print("Starting Baysian Gaussian Mixture Model...")
    bgm_acc_diag = bayesian_gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag")
    all_scores.append(mean_kfolds(bgm_acc_diag))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished Bayesian Gaussian Mixture Model.\n")

    # TAKES A LOT OF TIME FOR COMPLETE DATA SET
    # D + E -> different data preparation necessary

    # D label spreading model
    print("Starting Label Spreading Model...")
    ls_scores = label_spreading(x_train=x_train_all, y_train=y_train_all, cv_semisupervised=cv_semisupervised, name="LabelSpreading_1000")
    all_scores.append(mean_kfolds(ls_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished Label Spreading Model.\n")

    # E self training model
    print("Starting Self Training Classifier...")
    knn = KNeighborsClassifier(n_neighbors = 20, weights = "distance") # TODO replace with balanced random forest
    st_scores = self_training(x_train=x_train_all, y_train=y_train_all, cv_semisupervised=cv_semisupervised, base_model=knn, name="SelfTraining_1000")
    all_scores.append(mean_kfolds(st_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished Self Training Classifier.\n")

    # F random forests (works)
    print("Starting Random Forest Model ...")
    rf_scores = random_forest(x_train, y_train, cv_stratified, visualize=False)
    all_scores.append(mean_kfolds(rf_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished Random Forest Model.\n")

    # G Support Vector Machines
    # works with very high gamma (overfitting) -> "auto" yields 0.75, still good and no overfitting
    print("Starting Support Vector Machine...")
    svm_scores = svm(x_train, y_train, cv_stratified, gamma="auto")
    all_scores.append(mean_kfolds(svm_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished Support Vector Machine.\n")

    # H knn (works with weights=distance)
    print("Starting K-Nearest Neighbours Model...")
    knn_scores = knn(x_train, y_train, cv_stratified, n_neighbors=20)
    all_scores.append(mean_kfolds(knn_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished K-Nearest Neighbours Model.\n")

    # I adaboost
    print("Starting AdaBoost Model...")
    ada_scores = ada_boost(x_train, y_train, cv_stratified)
    all_scores.append(mean_kfolds(ada_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished AdaBoost Model.\n")

    # J LSTM
    print("Starting LSTM Model...")
    lstm_scores = ann(x_train, y_train, smp_idx_train, ann_type="lstm", cv=cv_timeseries, name="LSTM",
                      batch_size=32, epochs=10, rnn_size=25, dense_units=25, dropout=0.2, learning_rate=0.01)
    print(lstm_scores)
    all_scores.append(mean_kfolds(lstm_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished LSTM Model.\n")

    # K BLSTM
    print("Starting BLSTM Model...")
    #  cv can be a float, or a cv split
    blstm_scores = ann(x_train, y_train, smp_idx_train, ann_type="blstm", cv=cv_timeseries, name="BLSTM",
                       batch_size=32, epochs=10, rnn_size=25, dense_units=25, dropout=0.2, learning_rate=0.01)
    print(blstm_scores)
    all_scores.append(mean_kfolds(blstm_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished BLSTM Model.\n")

    # J Encoder-Decoder
    print("Starting Encoder-Decoder Model...")
    #  cv can be a float, or a cv split
    encdec_scores = ann(x_train, y_train, smp_idx_train, ann_type="enc_dec", cv=cv_timeseries, name="ENC_DEC",
                       batch_size=32, epochs=10, rnn_size=25, dense_units=0, dropout=0.2, learning_rate=0.001,
                       attention=True, bidirectional=False)
    print(encdec_scores)
    all_scores.append(mean_kfolds(encdec_scores))
    if intermediate_file is not None: save_results(intermediate_file, all_scores)
    print("...finished Encoder-Decoder Model.\n")

def main():
    args = parser.parse_args()
    #test = "data/preprocessed_data_test.txt"

    if args.preprocess:
        data = preprocess_dataset(smp_file_name=args.smp_npz, output_file=args.preprocess_file, visualize=False) #
    else:
        with open(args.preprocess_file, "rb") as myFile:
            data = pickle.load(myFile)

    # EVALUATION
    if args.evaluate: evaluate_all_models(data, overwrite_tables=False)

    # TRAINING AND STORING MODEL
    # models can be modified here (make a list of desired models, params
    # from BEST_PARAMS are used!)
    if args.train_and_store: train_and_store_models(data, models=args.models)

    # VALIDATION
    if args.validate:
        intermediate_results = "data/validation_results.txt"
        validate_all_models(data, intermediate_results)

        all_scores = load_results(intermediate_results)

        # print and save results the validation results
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
        print(tabulate(pd.DataFrame(all_scores), headers="keys", tablefmt="psql"))

        with open("output/tables/models_160smp_test01.txt", 'w') as f:
            f.write(tabulate(pd.DataFrame(all_scores), headers="keys", tablefmt="psql"))

        with open("output/tables/models_160smp_test01_latex.txt", 'w') as f:
            f.write(tabulate(pd.DataFrame(all_scores), headers="keys", tablefmt="latex_raw"))



if __name__ == "__main__":
    random.seed(42)
    main()
