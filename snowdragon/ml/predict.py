# get the necessary imports from folders above
import os
import git
import joblib
import pickle

import numpy as np
from tqdm import tqdm
from pathlib import Path
from tensorflow import keras
from itertools import groupby
from keras_self_attention import SeqSelfAttention

from snowdragon.ml.models.baseline import predict_baseline
from snowdragon.ml.evaluation.cv_handler import assign_clusters_single_profile
from snowdragon.ml.models.anns import predict_single_profile_ann
from snowdragon.ml.run_models import remove_nans_mosaic, normalize_mosaic
from snowdragon.utils.helper_funcs import int_to_idx
from tuning.tuning_parameters import BEST_PARAMS
from data_handling.data_parameters import ANTI_LABELS, PARAMS
from data_handling.data_preprocessing import export_pnt, npz_to_pd, search_markers

# make an argparser for knowing which model should be used
# and which files should be processed
# and where the results should be saved (and if visualizations should be stored as well)

# make predictions for all files in this folder

IN_DIR = "/home/julia/Documents/University/BA/Data/Arctic_updated/"

# save both the pics and the results as .ini files
def predict_profile(smp, model, data, model_type):
    """ Predict classification of single profile
    """
    train_data = data["x_train"]
    train_targets = data["y_train"]

    # prepare smp data for which want to create prediction
    x_unknown_profile = smp.drop(["label", "smp_idx"], axis=1)

    # predict on that model
    if model_type == "scikit": # XXX WORKS
        y_pred = model.predict(x_unknown_profile)
        # TODO does not work for svm --> maybe we stored the wrong model? (always the same prediction)

    elif model_type == "keras": # XXX WORKS
        # TODO won't work right now
        y_pred = predict_single_profile_ann(model, x_unknown_profile, train_targets)

    elif model_type == "baseline": # XXX WORKS
        majority_vote = model
        y_pred = predict_baseline(majority_vote, x_unknown_profile)

    elif model_type == "semi_manual": # XXX WORKS
        # determine the number of clusters/components
        if hasattr(model, "cluster_centers_"):
            num_components = model.cluster_centers_.shape[0]
        elif hasattr(model, "weights_"):
            num_components = model.weights_.shape[0]

        # prediction -> data points are assigned to clusters from training!
        pred_clusters = model.predict(x_unknown_profile)
        train_clusters = model.predict(train_data)
        # find out which cluster is which label!
        y_pred = assign_clusters_single_profile(pred_clusters, train_targets, train_clusters, num_components)

    else:
        raise ValueError("""This Model implementation types does not exist.
        Choose one of the following: \"scikit\" (for rf, rf_bal, svm, knn, easy_ensemble, self_trainer),
        \"semi_manual\" (for kmean, gmm, bmm), \"keras\" (for lstm, blsm, enc_dec),
        or \"baseline\" (for the majority vote baseline)""")

    return y_pred

def load_markers(marker_path):
    """ Loads and returns sfc and ground markers with profile name as key
    Parameters:
        marker_path (Path): where the markers are stored
    Returns:
        dict < smp_name: (sfc_marker, ground_marker) >: marker dictionary
    """
    # load markers
    with open(marker_path, 'r') as file:
        marker_dic = {}
        for line in file:
            line = line.strip('\n')
            (key, sfc_val, ground_val) = line.split(',')
            marker_dic[key] = (float(sfc_val), float(ground_val))
    return marker_dic

def predict_all(unlabelled_dir=IN_DIR, marker_path="data/sfc_ground_markers.csv", mm_window=1, overwrite=True):
    """ Main function to predict the given set of profiles
    Parameters:
        unlabelled_dir (Path): where the unlabelled data is stored
        marker_path (Path): csv file with sfc and ground markers
        mm_window (int or float): default: one unit represents 1 mm. Choose the value
            you have used during data-preprocessing. Default value is 1mm there as well.
        overwrite (bool): default = False means if the file exists it is not overwritten but skipped
    """
    # TODO make this a function parameter
    location = "output/predictions/"
    Path(location).mkdir(parents=True, exist_ok=True)

    # we need some of the information from our training data
    with open("data/preprocessed_data_k5.txt", "rb") as handle:
        data = pickle.load(handle)

    # get current git commit
    repo = git.Repo(search_parent_directories=True)
    git_id = repo.head.object.hexsha
    # baseline ALL
    # lstm ALL
    # rf ALL
    # rf_bal ALL
    # svm SOME
    # knn ALL
    # easy_ensemble SOME
    # self_trainer ALL
    # kmeans ALL
    # gmm ALL
    # bmm ALL
    # blstm ALL
    # enc_dec ALL
    # "baseline", "rf", "rf_bal", "knn",
    models = ["kmeans", "gmm", "bmm", "lstm", "blstm", "enc_dec", "self_trainer", "easy_ensemble"]
    models = ["svm"]
    # TODO svm
    smp_profiles = load_profiles(unlabelled_dir)

    markers = load_markers(marker_path)

    # for all desired models create predictions
    for model_name in models:
        print("Starting to create predictions for model {}:".format(model_name))
        sub_location = location + "/" + model_name + "/"
        # make dir if it doesnt exist yet
        if not os.path.exists(sub_location):
            os.makedirs(sub_location)

        # load model
        model, model_type = load_stored_model(model_name)

        # for all desired data create model predictions
        for unlabelled_smp in tqdm(smp_profiles):
            # reset index for unlabelled_smp
            unlabelled_smp.reset_index(inplace=True, drop=True)
            # get smp idx
            smp_idx_str = int_to_idx(unlabelled_smp["smp_idx"][0])
            save_file = sub_location + "/" + smp_idx_str + ".ini"

            # predict profile
            if (not Path(save_file).is_file()) or overwrite:
                # fill nans
                unlabelled_smp.fillna(method="ffill", inplace=True)
                # if only nans, ffill won't work, but in this case: skip profile
                if sum(unlabelled_smp.isnull().any(axis=1)) == 0:

                    labelled_smp = predict_profile(unlabelled_smp, model, data, model_type)

                    try: # get markers
                        sfc, ground = markers[smp_idx_str]
                        # save ini
                        save_as_ini(labelled_smp, sfc, ground, save_file, model_name, git_id)
                    except KeyError:
                        print("Skipping Profile " + smp_idx_str + " since it is not contained in the marker file.")
                    # save figs
                    #save_as_pic(labelled_smp)
                else:
                    print("Skipping Profile "+ smp_idx_str + " since it contains to many NaNs.")


def save_as_ini(labelled_smp, sfc, ground, location, model, git_commit_id, mm_window=1):
    """ Save the predictions of an smp profile as ini file.
    Parameters:
        labelled_smp (list): predictions of a single unknown smp profile
        sfc (float): marker where the profiles surface originally was
        ground (float): marker where the profiles ground originally was
        location (str): where to save the ini file. Could be output/predictions/
            or within a file directory
        model (str): which model was used to generate the prediction
        mm_window (int or float): default: one unit represents 1 mm. Choose the value
            you have used during data-preprocessing. Default value is 1mm there as well.
    """
    # find out how ini files look like
    # .ini files are simple text files
    # distance values of the labelled smp file
    dist_list = [dist * mm_window for dist in range(len(labelled_smp))]
    # move window over labelled smp

    labels_occs = [(key, sum(1 for i in group)) for key, group in groupby(labelled_smp)]
    str_label_dist_pairs = [] # (label_str, last dist point where label occurs (inklusive!))
    complete_dist = 0

    # label_occ: (label, number of consecutive occurences)
    for label_occ in labels_occs:
        str_label = ANTI_LABELS[label_occ[0]]
        if complete_dist == 0:
            dist = (label_occ[1] - 1 + sfc) * mm_window
        else:
            dist = (label_occ[1]) * mm_window
        complete_dist += dist
        str_label_dist_pairs.append((str_label, complete_dist))

    with open(location, 'w') as file:
        file.write("[markers]\n")
        # file.write("# [model] = " + model) # model must be included in function
        # add surface marker
        file.write("surface = " + str(sfc) + "\n")
        # add snowgrain markers
        for (label, dist) in str_label_dist_pairs:
            file.write(label + " = " + str(dist) + "\n")
        # ground level is the same like last label
        file.write("ground = " + str(ground) + "\n")
        file.write("[model]\n")
        file.write(model + " = " + git_commit_id)

def load_profiles(data_dir, overwrite=False):
    """
    Returns:
        list <pd.Dataframe>: normalized smp data
    """
    #export_dir = Path("data/smp_profiles_unlabelled/")
    export_dir = Path("data/smp_profiles_updated/")
    data_dir = Path(data_dir)
    marker_path = Path("data/sfc_ground_markers.csv")
    export = False
    markers = False
    filter = True
    # export data from pnt to csv or npz
    if export:
        export_pnt(pnt_dir=data_dir, target_dir=export_dir, export_as="npz", overwrite=False, **PARAMS)
    if markers:
        search_markers(pnt_dir=data_dir, store_dir=marker_path)

    # load_data(npz_name, test_print=False, **kwargs)
    # load pd.DataFrame from all npz files and save this pd as united DataFrame in npz
    all_smp = npz_to_pd(export_dir, is_dir=True)

    # Filter all profiles out that are already labelled
    if filter:
        # unlabelled data
        all_smp = all_smp[(all_smp["label"] == 0)]

    # normalize the data to get correct predictions
    all_smp = normalize_mosaic(all_smp)

    # create list structure of smp profiles
    num_profiles = all_smp["smp_idx"].nunique()
    num_points = all_smp["smp_idx"].count()
    idx_list = all_smp["smp_idx"].unique()
    smp_list = [all_smp[all_smp["smp_idx"] == idx] for idx in idx_list]

    print("Finished Loading " + str(len(smp_list)) + " Profiles")

    return smp_list

def load_stored_model(model_name):
    """
    """
    # find out which model_type we have
    if model_name == "baseline":
        model_type = "baseline"
    elif model_name == "bmm" or model_name == "gmm" or model_name == "kmeans":
        model_type = "semi_manual"
    elif model_name in ["easy_ensemble", "knn", "label_spreading", "self_trainer", "rf", "rf_bal", "svm"]:
        model_type = "scikit"
    elif model_name == "lstm" or model_name == "blstm" or model_name == "enc_dec":
        model_type = "keras"
    else:
        raise ValueError("The model you have chosen does not exist in the list of stored models. Consider running and storing it.")

    # get stored model (models/stored_models)
    if model_type == "keras":
        model_filename = "models/stored_models/" + model_name + ".hdf5"
        if model_name != "enc_dec":
            loaded_model = keras.models.load_model(model_filename)
        else:
            loaded_model = keras.models.load_model(model_filename, custom_objects={"SeqSelfAttention": SeqSelfAttention})
    else:
        model_filename = "models/stored_models/" + model_name + ".model"
        with open(model_filename, "rb") as handle:
                if model_name == "rf":
                    loaded_model = joblib.load(handle)
                else:
                    loaded_model = pickle.load(handle)

    return loaded_model, model_type

def make_dirs():
    """
    """
    pass

def save_as_pic():
    """
    """
    # TODO make function for afterwards instead!!!
    # generate picture only for specific profiles
    pass



if __name__ == "__main__":
    predict_all()
