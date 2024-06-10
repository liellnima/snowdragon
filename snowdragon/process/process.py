import os
import glob
import time
import pickle 
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from snowmicropyn import Profile
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold


from snowdragon.ml.evaluation.cv_handler import cv_manual
from snowdragon.utils.helper_funcs import load_smp_data, load_configs, npz_to_pd, idx_to_int
from snowdragon.ml.evaluation.train_test_splitter import my_train_test_split
from snowdragon.visualize.visualize import visualize_original_data, visualize_normalized_data
from snowdragon.process.process_profile_funcs import label_pd, relativize, remove_negatives, summarize_rows, rolling_window
from snowdragon.process.process_dataset_funcs import remove_nans_mosaic, normalize_dataset, mosaic_specific_processing


# exports pnt files (our smp profiles!) to csv files in a target directory
def preprocess_all_profiles(
        data_dir: Path, 
        export_dir: Path, 
        labels: dict,
        npz_name: str = "smp_all.npz", 
        export_as: str = "npz", 
        overwrite: bool = False, 
        **params,
    ):
    """ Exports all pnt files from a dir and its subdirs as csv files into a new dir.
    Preproceses the profiles, according to kwargs arguments.
    Parameters:
        data_dir (Path): Directory where all the pnt data (= the smp profiles) and their ini files (= the smp markers) are stored
        export_dir (Path): Directory where the exported npz files (one per each smp profile) is saved
        labels (dir): Dictionary of all labels (name: int) used in the dataset
        npz_name (String): how the npz file should be called where all the smp profiles are stored together (must end with .npz)
        overwrite (Boolean): Default False. If the smp profiles were already exported once into the export_dir,
            this data can be overwritting by setting overwrite = True. Otherwise only those files will be exported that do not exist yet.
        export_as (String): either as "csv" or "npz" (default)
        **kwargs:
            sum_mm (num): How many mm of the smp data should be summed up together? (e.g. 1 -> resolution of 1mm snow layers)
            gradient (Boolean): Should the gradient be included in the formed dataset?
            window_size (list): Window sizes that should be applied during preprocessing
            window_type (String): Window type of the rolling window
            window_type_std (int): std used for window type
            rolling_cols (list): List of columns over which should be rolled
            poisson_cols (list): List of features that should be taken from poisson shot model.
                List can include: "distance", "median_force", "lambda", "f0", "delta", "L"
    
    """
    print("Starting to export and/or convert data")
    start = time.time()

    ### EXPORT DATA TO PNT ###########
    # create dir for csv exports
    if not export_dir.is_dir():
        export_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        raise ValueError("The following directory does not exist: {}. \nPlease use a directory in the main configs for raw_data, smp that contains pnt files.".format(data_dir))

    # match all files in the dir who end on .pnt recursively
    match_pnt = data_dir.as_posix() + "/**/*.pnt"
    # use generator to reduce memory usage
    file_generator_size = len(list(glob.iglob(match_pnt, recursive=True)))
    file_generator = glob.iglob(match_pnt, recursive=True)

    # yields each matching file and exports it
    #print("Progressbar is only correct when using the Mosaic SMP Data.")
    with tqdm(total=file_generator_size) as pbar: #3825
        for file in file_generator:
            file_name = Path(export_dir, file.split("/")[-1].split(".")[0] + "." + export_as)
            # exports file only if we want to overwrite it or it doesnt exist yet
            if overwrite or not file_name.is_file():
                smp_profile = Profile.load(file)
                # indexes, labels, summarizes and applies a rolling window to the data
                preprocess_profile(smp_profile, export_dir, labels, export_as=export_as, **params)
            pbar.update(1)

    print("Finished exporting all pnt file as {} files in {}.".format(export_as, export_dir))
    ##########################

    # load pd.DataFrame from all npz files and save this pd as united DataFrame in npz
    smp_first = npz_to_pd(export_dir, is_dir=True)
    dict = smp_first.to_dict(orient="list")
    np.savez_compressed(npz_name, **dict)

    end = time.time()
    print("Elapsed time for export and dataframe creation: ", end-start)
    print("\nUnited smp data was stored in {}.".format(npz_name))



def preprocess_profile(
        profile: Profile, 
        target_dir: Path, 
        labels: dict,
        export_as: str ="csv", 
        sum_mm: float = 1.0, 
        gradient: bool = False, 
        **params,
    ):
    """ Preprocesses a smp profile. Jobs done:
    Indexing, labelling, select data between surface and ground (and relativizes this data).
    Summarizing data in a certain mm window (reduces precision/num of rows).
    Applies a rolling window.
    Exports profile as csv.

    Parameters:
        profile (Profile): the profile which is preprocessed
        target_dir (Path): in which directory the data should be saved
        labels (dict): labels in form of a dictionary
        export_as (String): how the data should be exported. Either as "csv" possible or "npz"
        sum_mm (num): arg for summarize_rows function - indicates how many mm should be packed together
        gradient (Boolean): arg to decide whether gradient of each datapoint should be calculated
        **params:
            window_size (list): arg for rolling_window function - List of window sizes that should be applied. e.g. [4]
            rolling_cols (list): arg for rolling_window function - List of columns over which should be rolled
            window_type (String): arg for rolling_window function - E.g. Gaussian (default). None is a normal window.
            window_type_std (int): arg for rolling_window function - std used for window type
            poisson_cols (list): arg for rolling_window function - List of features that should be taken from poisson shot model
                List can include: "distance", "median_force", "lambda", "f0", "delta", "L"
    """

    # check if this is a labelled profile
    labelled_data = len(profile.markers) != 0

    # detect surface and ground if labels don't exist yet
    if not labelled_data:
        # no markers exist yet
        try:
            profile.detect_ground()
            profile.detect_surface()
        except ValueError:
            print("Profile {} is too short for data processing. Profile is skipped.".format(profile.name))
            # leave function
            return

    # 0. Store ground and surface labels for the profile
    # with open("data/sfc_ground_markers.csv", "a+") as file:
    #     writer = csv.writer(file)
    #     writer.writerow([profile.name, profile.detect_surface(), profile.detect_ground()])

    # 1. restrict dataframe between surface and ground (absolute distance values!)
    df = profile.samples_within_snowpack(relativize=False)

    # add label column
    df["label"] = 0

    # 2. label dataframe, if labels are there
    if labelled_data:
        label_pd(df, profile, labels)

    # 3. relativize, such that the first distance value is 0
    relativize(df)

    # 4. Remove all values below 0, replace them with average value around
    if any(df["force"] < 0):
        # skip profile if more than 80 % of the values are negative
        if sum(df["force"] < 0) >= (0.8 * len(df)):
            print("Profile {} contains more than 80% negative values. Profile is skipped.".format(profile.name))
            return
        df = remove_negatives(df, col="force")

    # 5. summarize data (precision: 1mm)
    df_mm = summarize_rows(df, mm_window=sum_mm)

    # 6. rolling window in order to know distribution of next and past values (+ poisson shot model)
    final_df = rolling_window(df_mm, **params)

    # 7.include gradient if wished
    if gradient:
        try:
            final_df["gradient"] = np.gradient(final_df["mean_force"])
        except ValueError as e:
            if len(e.args) > 0 and e.args[0] == "Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.":
                print("Array was too small, replaced gradient with 0.")
                final_df["gradient"] = 0
            else:
                raise e

    # 8. add other features, index DataFrame and convert dtypes
    # Add SMP idx
    final_df["smp_idx"] = idx_to_int(profile.name)
    # Add relative position feature, if df is not empty
    final_df["pos_rel"] = final_df.apply(lambda x: x["distance"] / final_df["distance"].max(), axis=1) if not final_df.empty else 0
    # Add distance from ground, if df is not empty
    final_df["dist_ground"] = final_df.apply(lambda x: final_df["distance"].max() - x["distance"], axis=1) if not final_df.empty else 0


    for col in final_df:
        if col == "label" or col == "smp_idx":
            final_df[col] = final_df[col].astype("int32")
        else:
            final_df[col] = final_df[col].astype("float32")

    # export as csv or npz
    if export_as == "csv":
        final_df.to_csv(os.path.join(target_dir, Path(profile.name + ".csv")))
    elif export_as == "npz":
        dict = final_df.to_dict(orient="list")
        np.savez_compressed(os.path.join(target_dir, Path(profile.name)), **dict)
    else:
        raise ValueError("export_as must be either csv or npz")


def preprocess_dataset(
        smp_file_name, 
        smp_normalized_file_name, 
        output_file = None, 
        random_seed: int = 42, 
        visualize: bool = False, 
        sample_size_unlabelled: int =1000, 
        tsne: int = 0, 
        ignore_unlabelled: bool = False, 
        k_fold: int = 5,
        test_size: float = 0.2, 
        train_size: float = 0.8, 
        original_mosaic_dataset: bool = True,
        label_configs_name: str = "graintypes.yaml",
        visualize_configs_name: str = "visualize.yaml",
        **kwargs
    ):
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
        ignore_unlabelled (bool): If the unlabelled data should be ignored during preprocessing. Default is false.
    Returns:
        (dict): "x_train", "y_train", " x_test" and "y_test" are the prepared and normalized training and test data.
                "x_train_all" and "y_train_all" are both labelled and unlabelled data in one data set.
                "unlabelled_smp_x" is the complete unlabelled input data without any labelled data points.
                "cv", "cv_semisupervised" and "cv_timeseries" are cv splits for the supervised, semisupervised and ann models.
                "smp_idx_train" and "smp_idx_test" are the smp indiced corresponding to training and test data (correct order)
    """
    # 0. Load the relevant configs 
    visualize_configs = load_configs("visualize", visualize_configs_name)
    label_configs = load_configs("graintypes", label_configs_name)

    # 1. Load dataframe with smp data
    smp_org = load_smp_data(smp_file_name)

    # remove nans
    #TODO ADAPT THIS FUNCTION FOR YOUR OWN DATASET IF NEEDED
    if original_mosaic_dataset:
        smp_org = remove_nans_mosaic(smp_org)

    # 2. Visualize before normalization
    if visualize: visualize_original_data(
        smp=smp_org, 
        example_smp_name=visualize_configs["example_smp_name"], 
        labels=label_configs["label"]
        **visualize_configs["original"],
        )

    # 3. Normalize
    smp = normalize_dataset(smp_org)

    # CHANGE HERE - YOUR FAVORITE LABELS IN THE DATA!

    # 4. Sum up certain classes if necessary (alternative: do something to balance the dataset)
    if original_mosaic_dataset:
        smp = mosaic_specific_processing(smp, label_configs["labels"])

    # save normalized and pre-processed data
    dict = smp.to_dict(orient="list")
    np.savez_compressed(smp_normalized_file_name, **dict)

    # 5. Visualize the data after normalization
    if visualize: visualize_normalized_data(
        smp=smp,
        example_smp_name=visualize_configs["example_smp_name"],
        used_labels=label_configs["used_labels"] + label_configs["rare_labels"],
          **visualize_configs, **visualize_configs["normalize"] **label_configs
        )

    # if wished, make dimension reduction here!
    if tsne != 0:
        labels = smp["label"]
        indices = smp["smp_idx"]
        smp_x = smp.drop(["label", "smp_idx"], axis=1)
        tsne_model = TSNE(n_components=tsne, verbose=1, perplexity=40, n_iter=300, random_state=42)
        tsne_results = tsne_model.fit_transform(smp_x)
        smp_with_tsne = {"label": labels, "smp_idx": indices}

        for i in range(tsne):
            smp_with_tsne["tsne" + str(i)] = tsne_results[:, i]

        smp = pd.DataFrame(smp_with_tsne)

    print(smp)
    # 6. Prepare unlabelled data for two of the semisupervised modles:
    if ignore_unlabelled:
        unlabelled_smp_x = None
        unlabelled_x = None
        unlabelled_y = None
    else:
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
    x_train, x_test, y_train, y_test, smp_idx_train, smp_idx_test = my_train_test_split(smp, test_size=test_size, train_size=train_size)

    # For two of the semisupervised models: include unlabelled data points in x_train and y_train (test data stays the same!)
    x_train_all = None if ignore_unlabelled else pd.concat([x_train, unlabelled_x])
    y_train_all = None if ignore_unlabelled else pd.concat([y_train, unlabelled_y])

    # 8. Make crossvalidation split
    # Note: if we want to use StratifiedKFold, we can just hand over an integer to the functions
    cv_stratified = list(StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_seed).split(x_train, y_train))
    # Attention the cv fold for these two semi-supervised models is different from the other cv folds!
    cv_semisupervised = list(StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_seed).split(x_train_all, y_train_all))
    data = x_train.copy()
    target = y_train.copy()
    data["smp_idx"] = smp_idx_train
    cv_timeseries = cv_manual(
            data=data, 
            target=target, 
            k=k_fold, 
            random_state=random_seed
        )
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