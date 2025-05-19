# Data Preprocessing is done here
#from data_handling.data_parameters import SMP_LOC, T_LOC, EXP_LOC, LABELS, PARAMS
#from data_handling.data_parameters import SMP_ORIGINAL_NPZ
from snowdragon import DATA_DIR
from snowdragon.utils.idx_funcs import idx_to_int

# external imports
import os
import csv
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path # for os independent path handling
from snowmicropyn import Profile, loewe2012, windowing


def search_markers(
        pnt_dir: Path, 
        store_dir: Path = DATA_DIR / "sfc_ground_markers.csv",
    ):
    """ Detect and store all the surface and ground markers.
    Parameters:
        pnt_dir (Path): folder location of pnt files (in our case the smp profiles)
        store_dir (Path): folder location where marker csv should be stored. if
            the file already exists the data is simply appended
    """
    # match all files in the dir who end on .pnt recursively
    match_pnt = pnt_dir.as_posix() + "/**/*.pnt"
    # use generator to reduce memory usage
    file_generator_size = len(list(glob.iglob(match_pnt, recursive=True)))
    file_generator = glob.iglob(match_pnt, recursive=True)

    # yields each matching file and stores the markers data
    #print("Progressbar is only correct when using the Mosaic SMP Data.")
    with tqdm(total=file_generator_size) as pbar: # 3825
        for file in file_generator:
            profile = Profile.load(file)
            labelled_data = len(profile.markers) != 0
            # detect surface and ground if labels don't exist yet
            if not labelled_data:
                try:
                    profile.detect_ground()
                    profile.detect_surface()
                    # get surface and ground
                    with open(store_dir, "a+") as file:
                        writer = csv.writer(file)
                        writer.writerow([profile.name, profile.detect_surface(), profile.detect_ground()])
                except ValueError:
                    print("Profile {} is too short for data processing. Profile is skipped.".format(profile.name))

            pbar.update(1)

    print("Finished storing surface and ground markers in {}.".format(store_dir))


# function to get temperature data
def get_temperature(temp: Path) -> pd.DataFrame:
    """ Gets temperature data from files
    Parameters:
        temp (Path): file location of temperature data
    Returns:
        pd.DataFrame: complete data in pd.DataFrame format
    """
    return pd.read_csv(temp)

# method to check if all pnt files have found their way into the united smp dataframe
# ATTENTION: this method takes extremely long! Some minutes for checking the existence of one file.
# It works perfectly fine and fast for small dataframes. Searching large dataframes just takes time.
def check_export(
        pnt_dir: Path, 
        smp_df: pd.DataFrame,
        break_imm: bool = True,
    ) -> bool:
    """ Checks if all smp pnt files can be found in a dataframe. Takes a lot of time.
    Parameters:
        pnt_dir (Path): folder location where all the unlabelled pnt data is stored
        smp_df (pd.DataFrame): dataframe with column "smp_idx", where the complete smp data is collected
        break_imm (Boolean): indicates if search should be aborted immediately when a file was not found.
            This is faster and prints out which file has not been found. Default value is True.
    Returns:
        Boolean: True if all files can be found in the dataframe, False otherwise
    """
    # check if dir exists
    if not os.path.exists(pnt_dir):
        print("Targeted directory for pnt files does not exist.")
    # check if dir is empty
    if len(os.listdir(pnt_dir)) == 0:
        print("Warning: Targeted directory for pnt files is empty.")

    # stores whether file was found
    found_all = []

    # match all files in the dir who end on .pnt recursively
    match_pnt = pnt_dir.as_posix() + "/**/*.pnt"
    # use generator to reduce memory usage
    file_generator = glob.iglob(match_pnt, recursive=True)
    # yields each matching file and exports it
    for file in file_generator:
        print("Determining if file {} is in the dataframe".format(file))
        smp_was_found = any(smp_df.smp_idx == idx_to_int(file.split("/")[-1].split(".")[0]))
        found_all.append(smp_was_found)
        if break_imm and not smp_was_found:
            print("The following file was not found: ", file.split(".")[0])
            return False

    # if all values in found_all are True, return True
    if all(smp_found == True for smp_found in found_all):
        return True
    # in all other cases print how many where found and how many are missing
    print("Number of files found in the dataframe: ", found_all.count(True))
    print("Number of files NOT found in the dataframe: ", found_all.count(False))
    # and return False
    return False


def label_pd(df: pd.DataFrame, profile: Profile, labels: dict):
    """ Labels the given pandas dataframe, according to the markers saved in profile.
    Parameters:
        df (pd.DataFrame): the dataframe to label
        profile (Profile): the smp profile where the markers are saved (in a separate ini file)
        labels (dict): dictionary of labels
    """
    # save starting point of labelling
    last_marker = profile.markers.get("surface")
    # markers are assigned to the pd.DataFrame in a sorted manner
    for marker in sorted(profile.markers, key=profile.markers.get, reverse=False):
        if marker is not "surface" or "ground" or "not_labelled":
            # everything between last_marker and new marker gets labelled
            sel_rows = (df["distance"] > last_marker) & (df["distance"] <= profile.markers.get(marker))
            integer_label = labels.get(marker.translate({ord(ch): None for ch in '0123456789'}))
            if integer_label is None:
                raise ValueError("LABELS does not contain the marker {}. Please add it to continue.".format(marker))
            df.loc[sel_rows, "label"] = integer_label
            # assign new last_marker
            last_marker = profile.markers.get(marker)

def relativize(df: pd.DataFrame):
    """ Relativizes a dataframe with column "distance", such that the first distance value is 0.
    Parameters:
        df (pd.DataFrame): the dataframe whose distance column is relativized
    """
    surface_value = df["distance"].iloc[0]
    df["distance"] = df["distance"].apply(lambda x: x - surface_value)

def summarize_rows(
        df: pd.DataFrame, 
        mm_window: float = 1.0,
    ) -> pd.DataFrame:
    """ Summarizes the rows of a dataframe with columns "force", "distance" and "labels".
    Produces mean, var, min and max of force. Most often label is used. Last distance point is used.
    Parameters:
        df (pd.DataFrame): DataFrame that is summarized.
        mm_window (num): how many mm should be summed up. E.g. 1 [mm] (default), 0.5 [mm], etc.
    Returns:
        pd.DataFrame: the summarized DataFrame
    """
    # number of rows we summarize (1mm = 242 rows)
    window_size = mm_window * 242
    window_stepper = range(0, len(df)-window_size, window_size)

    # get stats for window
    mean_force = [df["force"].iloc[i:(i+window_size)].mean() for i in window_stepper]
    var_force = [df["force"].iloc[i:(i+window_size)].var() for i in window_stepper]
    min_force = [df["force"].iloc[i:(i+window_size)].min() for i in window_stepper]
    max_force = [df["force"].iloc[i:(i+window_size)].max() for i in window_stepper]
    distance = [df["distance"].iloc[i+window_size] for i in window_stepper]
    label = [df["label"].iloc[i:(i+window_size)].value_counts().idxmax() for i in window_stepper]

    # add the last data points which are cut off by the window stepper
    if len(window_stepper) > 0:
        # get range of lost data points
        lost_data = (window_stepper[-1] + window_size, len(df))

        # add last data points to all statistics
        mean_force.append(df["force"].iloc[lost_data[0] : lost_data[1]].mean())
        var_force.append(df["force"].iloc[lost_data[0] : lost_data[1]].var())
        min_force.append(df["force"].iloc[lost_data[0] : lost_data[1]].min())
        max_force.append(df["force"].iloc[lost_data[0] : lost_data[1]].max())
        # append correct next distance
        distance.append(distance[-1] + 1)
        # append correct last label
        label.append(df["label"].iloc[lost_data[0] : lost_data[1]].value_counts().idxmax())

    # returns summarized dataframe
    return pd.DataFrame(np.column_stack([distance, mean_force, var_force, min_force, max_force, label]),
                        columns=["distance", "mean_force", "var_force", "min_force", "max_force", "label"])


def rolling_window(
        df: pd.DataFrame, 
        window_size: list, 
        rolling_cols: list, 
        window_type:str = "gaussian", 
        window_type_std: int = 1, 
        poisson_cols: list = None, 
        **kwargs,
    ) -> pd.DataFrame:
    """ Applies one or several rolling windows to a dataframe. Concatenates the different results to a new dataframe.
    Parameters:
        df (pd.DataFrame): Original dataframe over whom we roll.
        window_size (list): List of window sizes that should be applied. e.g. [4]
        rolling_cols (list): list of columns over which should be rolled
        window_type (String): E.g. Gaussian (default). None is a normal window. Accepts any window types listed here:
            https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows
        window_type_std (int): std used for window type
        poisson_cols (list): List with names what should be retrieved from the poisson shot model. Default None (nothing included).
            List can include: "distance", "median_force", "lambda", "f0", "delta", "L"
        **kwargs: are catched in case more arguments are given on the commandline (see data_loader)
    Returns:
        pd.DataFrame: concatenated dataframes (original and new rolled ones)
    """
    all_dfs = [df]
    # for the poisson shot model calculations: df can only have cols force and distance
    poisson_df = pd.DataFrame(df[["distance", "mean_force"]])
    poisson_df.columns = ["distance", "force"]
    poisson_all_cols = ["distance", "median_force", "lambda", "f0", "delta", "L"]

    # roll over columns with different window sizes
    for window in window_size:
        # Roll window over 1mm summarizes statistics
        try:
            # window has the current distance point as center
            # for the first data points, only a few future datapoints will be used (min_periods=1 -> no na values)
            df_rolled = df[rolling_cols].rolling(window, win_type=window_type, center=True, min_periods=1).mean(std=window_type_std)
        except KeyError:
            print("The dataframe given does not have the columns indicated in rolling_cols.")
        # rename columns of rolled dataframe for distinction
        df_rolled.columns = [col + "_" + str(window) for col in rolling_cols]
        all_dfs.append(df_rolled)

        # Roll window for a poisson shot model to get lambda and delta
        if poisson_cols is not None:
            try:
                # calculate lambda and delta and media of poisson shot model
                overlap = (((window - 1) / window) * 100) + 0.0001 # add epsilon to round up 0.0001
                poisson_rolled = calc(poisson_df, window=window, overlap=overlap) #essentially the loewe2012.calc function
                poisson_rolled.columns = poisson_all_cols
                poisson_rolled = poisson_rolled[poisson_cols]
            except KeyError:
                print("You can only use a (sub)list of the following features for poisson_cols: distance, median_force, lambda, f0, delta, L")
            # add the poisson data to the all_dfs list and rename columns for distinction
            poisson_rolled.columns = [col + "_" + str(window) for col in poisson_cols]
            all_dfs.append(poisson_rolled)

    return pd.concat(all_dfs, axis=1)


# author Henning Loewe (2012) -> only difference: I am preventing zero divisions
def calc(
        samples: pd.DataFrame, 
        window: int, 
        overlap: float,
    ) -> pd.DataFrame:
    """Calculation of shot noise model parameters.
    :param samples: A pandas dataframe with columns called 'distance' and 'force'.
    :param window: Size of moving window.
    :param overlap: Overlap factor in percent.
    :return: Pandas dataframe with the columns 'distance', 'force_median',
             'L2012_lambda', 'L2012_f0', 'L2012_delta', 'L2012_L'.
    """
    # Calculate spatial resolution of the distance samples as median of all step sizes.
    spatial_res = np.median(np.diff(samples.distance.values))

    # Split dataframe into chunks
    chunks = windowing.chunkup(samples, window, overlap)
    result = []
    for center, chunk in chunks:
        f_median = np.median(chunk.force)
        # check if all elements are zero -> if yes, replace results with 0
        if all(item == 0 for item in chunk.force):
            sn = (0, 0, 0, 0) # a tuple containing four zeros (lamda, f0, delta, L)
        else:
            sn = loewe2012.calc_step(spatial_res, chunk.force)
        result.append((center, f_median) + sn)

    return pd.DataFrame(result, columns=['distance', 'force_median', 'L2012_lambda', 'L2012_f0',
                                         'L2012_delta', 'L2012_L'])

def remove_negatives(
        df: pd.DataFrame, 
        col: str = "force", 
        threshold: int =  -1
    ) -> pd.DataFrame:
    """ Remove negative values of a column from dataframe. The values are replaced with the mean of the next and last positive value.
    Parameters:
        df (pd.DataFrame): dataframe
        col (String): column which negative values should get replaced. Default: "force"
        threshold (int): Below which threshold the values should not just get assigned with 0?
    Returns:
        pd.DataFrame: where the negative values are removed
    """
    df_removed = df.copy(deep=True)
    # in case the values are just slightly below zero, replace them with 0
    df_removed.loc[(df_removed[col] < 0) & (df_removed[col] > threshold), col] = 0
    # if nothing is below a certain threshold, return result
    if not any(df_removed[col] < threshold):
        return df_removed

    # in all other cases replace the negative value with the mean of its next two neighbouring positive values
    index = pd.Series(df_removed.index)
    values_neg = df_removed[df_removed[col] < 0]
    indices_neg = index[df_removed[col] < 0]

    # for each negative value
    for value, index in zip(values_neg[col], indices_neg):
        # find the next positive value
        next_pos_idx = index
        # as long as index+1 is still in the list, add 1
        while (next_pos_idx in indices_neg) and (next_pos_idx < len(df_removed) - 1):
            next_pos_idx = next_pos_idx + 1
        next_pos_value = df_removed[col].iloc[next_pos_idx] if (next_pos_idx != len(df_removed)) else 0

        # find the last positive value
        last_pos_idx = index
        # as long as index-1 is still in the list, sub 1
        while (last_pos_idx in indices_neg) and (last_pos_idx > 0):
            last_pos_idx = last_pos_idx - 1
        last_pos_value = df_removed[col].iloc[last_pos_idx] if (last_pos_idx != 0) else 0

        # replace current value with mean of both
        df_removed.loc[index, col] = (next_pos_value + last_pos_value) / 2

    return df_removed
