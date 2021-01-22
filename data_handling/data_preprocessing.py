# Data Preprocessing is done here

# external imports
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import glob
import csv
import time
import re
import xarray
from pathlib import Path # for os independent path handling
from snowmicropyn import Profile, loewe2012, windowing
from scipy import signal
#from smpfunc import preprocess

# Set folder location of smp data (pnt format)
SMP_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/")
# Set file location of temperature data
T_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/MOSAiC_ICE_Temperature.csv")
# Set folder name were export files get saved
EXP_LOC = Path("smp_csv_test04")
# labels for the different grain type markers
LABELS = {"not_labelled": 0, "surface": 1, "ground": 2, "dh": 3, "dhid": 4, "mfdh": 5, "rgwp": 6,
          "df": 7, "if": 8, "ifwp": 9, "sh": 10, "drift_end": 11, "snow-ice": 12}
# arguments for Preprocessing
PARAMS = {"sum_mm": 1, "gradient": True, "window_size": [4,12], "window_type": "gaussian",
          "window_type_std": 1, "rolling_cols": ["mean_force", "var_force", "min_force", "max_force"],
          "poisson_cols": ["median_force", "lambda", "delta"]}

# exports pnt files (our smp profiles!) to csv files in a target directory
def export_pnt (pnt_dir, target_dir, export_as="npz", overwrite=False, **kwargs):
    """ Exports all pnt files from a dir and its subdirs as csv files into a new dir.
    Preproceses the profiles, according to kwargs arguments.
    Parameters:
        pnt_dir (Path): folder location of pnt files (in our case the smp profiles)
        target_dir (Path): folder name where converted csv files should get exported or were they have already been exported
        export_as (String): either as "csv" or "npz" (default)
        overwrite (Boolean): indicates if csv file should be overwriting if csv file already exists
        **kwargs: arguments for preprocessing function, for description see preprocess_profile()
    """
    # create dir for csv exports
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # match all files in the dir who end on .pnt recursively
    match_pnt = pnt_dir.as_posix() + "/**/*.pnt"
    # use generator to reduce memory usage
    file_generator = glob.iglob(match_pnt, recursive=True)
    # yields each matching file and exports it
    for file in file_generator:
        file_name = Path(target_dir, file.split("/")[-1].split(".")[0] + "." + export_as)
        # exports file only if we want to overwrite it or it doesnt exist yet
        if overwrite or not file_name.is_file():
            smp_profile = Profile.load(file)
            # indexes, labels, summarizes and applies a rolling window to the data
            preprocess_profile(smp_profile, target_dir, export_as=export_as, **kwargs)

    print("Finished exporting all pnt file as {} files in {}.".format(export_as, target_dir))

# TODO remove method
# unites the csv data into one csv and adds smp index to data
# and saves everything in pd.DataFrame that can be reloaded from the npz data
def get_smp_data(csv_dir, csv_filename="smp_all.csv", npz_filename=None, skip_unify=False, skip_npz=False):
    """ Gets unlabelled smp data. Indexes the data. Writes the complete data into one csv.
    csv files are subsequently transformed to pandas frame. Return the pd.DataFrame and stores it as npz.

    Parameters:
        csv_dir (Path): folder location of smp profiles as csv files
        csv_filename (String): Name for file, where all the smp data will be or is written to. File extension must be csv!
        npz_filename (String): File where npz data will be or is saved. If none (default) same name as csv_filename but with npz extension.
        skip_unify (Boolean): Default False. Set to True if csv_filename has already unified data in it.
        skip_npz (Bollean): Default False. Set to True if npz_filename has already stored data in it.
    """
    # check if something has been exported in smp_csv
    if len(os.listdir(csv_dir)) == 0:
        print("Your target directory is empty. Consider using pnt_to_csv first.")

    if not skip_unify:
        # unify all csv files into one and index the data with their SMP tag
        unify_and_index(csv_dir, csv_filename)

    if not skip_npz:
        # save csv file as npz
        save_csv_as_npz(csv_filename, npz_filename)

    # convert the npz to a pandas DataFrame and return it
    return npz_to_pd(npz_filename)


# TODO remove
def unify_and_index(csv_dir, csv_filename):
    """ Gets unlabelled smp data. Indexes the data. Writes the complete data into one csv.
    Parameters:
        csv_dir (Path): folder location of smp profiles as csv files
        csv_filename (String): Name for file, where all the smp data is written to. File extension must be csv!
    """
    # dictionary to resolve smp indexing
    smp_idx_resolver = {}

    # a list to save them all
    smp_all_rows = []
    # matching csv files recursively (recursively just to be safe)
    match_csv = csv_dir.as_posix() + "/**/*.csv"
    # use generator to reduce ram usage
    file_generator = glob.iglob(match_csv, recursive=True)

    # column names in csv files
    col_names = ["distance", "force", "smp_idx"]

    # we will write all data in one csv file. file is cleared automatically if already existant
    with open(csv_filename, "w+") as smp_all:
        # writer for writing rows in our file
        writer = csv.writer(smp_all)
        # yields each matching csv file
        for file in file_generator:
            # get smp datapoint name and convert it to int
            current_smp_idx = idx_to_int(file.split("/")[-1].split(".")[0])
            # open csv file and write each row to  smp_all_rows
            with open(Path(file)) as csv_file:
                # skip the first two lines of the file (comments)
                next(csv_file)
                next(csv_file)
                # read the rows of the current smp file (read as dictionaries!)
                current_smp_rows = csv.DictReader(csv_file, fieldnames=col_names)
                # each dictionary row is appended to the shared row list
                for row in current_smp_rows:
                     # write the row into the csv file
                     writer.writerow([row["distance"], row["force"], row["smp_idx"]])

    print("\nExported united csv files to {}.".format(csv_filename))

# TODO remove
def save_csv_as_npz(csv_filename, npz_filename):
    """ Exports a smp csv file produced by get_smp_data as npz.
    Paramters:
        csv_filename (String): The csv file with all smp profiles. Must have columns "distance", "force" and "smp_idx"
        npz_filename (String): Name of output npz file
    """
    # read each column as pd.DataFrame
    print("Progress 0/3: Start reading csv cols as pd.DataFrames")
    distance = pd.read_csv(csv_filename, usecols=[0], dtype=np.float32, sep=",", header=None, engine="c", low_memory=True)
    print("Progress 1/3: Read distance")
    force = pd.read_csv(csv_filename, usecols=[1], dtype=np.float32, sep=",", header=None, engine="c", low_memory=True)
    print("Progress 2/3: Read force")
    smp_idx = pd.read_csv(csv_filename, usecols=[2], dtype=np.int32, sep=",", header=None, engine="c", low_memory=True)
    print("Progress 3/3: Read index")

    # save columns as npz
    if npz_filename is None:
        npz_filename = csv_filename.split(".")[0] + ".npz"

    # save npz
    np.savez_compressed(npz_filename, distance=distance.values[:, 0], force=force.values[:, 0], smp_idx=smp_idx.values[:, 0])
    print("\nExported csv as numpy arrays to {}.".format(npz_filename))

def idx_to_int(string_idx):
    """ Converts a string that indexes the smp profile to an int.
    Paramters:
        string_idx (String): the index that is converted
    Returns:
        int32: the index as int.
        For smp profiles starting with S31H, S43M, S49M [1, 2, 3, 4] + the last four digits are the int.
        For smp profiles starting with PS122, [0] + 1 digit Leg + 2 digit week + 3 digit id are the int.
        All other profiles are 0.
    """
    if "PS122" in string_idx:
        str_parts = re.split("_|-", string_idx)
        #     Mosaic + Leg          + week                  + id number
        return int("1" + str_parts[1] + str_parts[2].zfill(2) + str_parts[3].zfill(3))

    elif "S31H" in string_idx:
        return int("2" + string_idx[-4:].zfill(6))
    elif "S43M" in string_idx:
        return int("3" + string_idx[-4:].zfill(6))
    elif "S49M" in string_idx:
        return int("4" + string_idx[-4:].zfill(6))
    else:
        return 0

# function to get temperature data
def get_temperature(temp):
    """ Gets temperature data from files
    Parameters:
        temp (Path): file location of temperature data
    Returns:
        dd.DataFrame: complete data in pd.DataFrame format
    """
    return pd.read_csv(temp)

# method to check if all pnt files have found their way into the united smp dataframe
# ATTENTION: this method takes extremely long! Some minutes for checking the existence of one file.
# It works perfectly fine and fast for small dataframes. Searching large dataframes just takes time.
def check_export(pnt_dir, smp_df, break_imm=True):
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

def print_test_df(smp):
    """ Printing some features and information of smp DataFrame.
    Paramters:
        smp (pd.DataFrame): dataframe from which the information is retrieved
    """
    print("Overview of smp dataframe: \n", smp.head())
    print("Info about smp dataframe:\n")
    smp.info()
    print("Dataypes of columns: \n", smp.dtypes)
    print("Datapoints per SMP File: \n", smp["smp_idx"].value_counts())
    print("First row: \n", smp.iloc[0])
    print("Force at first row: \n", smp[["mean_force", "var_force", "min_force", "max_force"]].iloc[0])
    print("Amount of datapoints with a force > 40: ", len(smp[smp["max_force"] > 40]))
    print("Was S31H0117 found in the dataframe? ", any(smp.smp_idx == idx_to_int("S31H0117")))
    print("Only S31H0117 data: \n", smp[smp["smp_idx"] == idx_to_int("S31H0117")].head())


def label_pd(df, profile):
    """ Labels the given pandas dataframe, according to the markers saved in profile.
    Parameters:
        df (pd.DataFrame): the dataframe to label
        profile (Profile): the smp profile where the markers are saved (in a separate ini file)
    """
    # save starting point of labelling
    last_marker = profile.markers.get("surface")
    # markers are assigned to the pd.DataFrame in a sorted manner
    for marker in sorted(profile.markers, key=profile.markers.get, reverse=False):
        if marker is not "surface" or "ground" or "not_labelled":
            # everything between last_marker and new marker gets labelled
            sel_rows = (df["distance"] > last_marker) & (df["distance"] <= profile.markers.get(marker))
            integer_label = LABELS.get(marker.translate({ord(ch): None for ch in '0123456789'}))
            if integer_label is None:
                raise ValueError("LABELS does not contain the marker {}. Please add it to continue.".format(marker))
            df.loc[sel_rows, "label"] = integer_label
            # assign new last_marker
            last_marker = profile.markers.get(marker)

def relativize(df):
    """ Relativizes a dataframe with column "distance", such that the first distance value is 0.
    Parameters:
        df (pd.DataFrame): the dataframe whose distance column is relativized
    """
    surface_value = df["distance"].iloc[0]
    df["distance"] = df["distance"].apply(lambda x: x - surface_value)

def summarize_rows(df, mm_window=1):
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
    # TODO do not cut off last part!
    window_stepper = range(0, len(df)-window_size, window_size)
    # get stats for window
    mean_force = [df["force"].iloc[i:(i+window_size)].mean() for i in window_stepper]
    var_force = [df["force"].iloc[i:(i+window_size)].var() for i in window_stepper]
    min_force = [df["force"].iloc[i:(i+window_size)].min() for i in window_stepper]
    max_force = [df["force"].iloc[i:(i+window_size)].max() for i in window_stepper]
    distance = [df["distance"].iloc[i+window_size] for i in window_stepper]
    label = [df["label"].iloc[i:(i+window_size)].value_counts().idxmax() for i in window_stepper]
    # returns summarized dataframe
    return pd.DataFrame(np.column_stack([distance, mean_force, var_force, min_force, max_force, label]),
                        columns=["distance", "mean_force", "var_force", "min_force", "max_force", "label"])

def rolling_window(df, window_size, rolling_cols, window_type="gaussian", window_type_std=1, poisson_cols=None):
    """ Applies one or several rolling windows to a dataframe. Concatenates the different results to a new dataframe.
    Parameters:
        df (pd.DataFrame): Original dataframe over whom we roll.
        window_size (list): List of window sizes that should be applied. e.g. [4]
        rolling_cols (list): list of columns over which should be rolled
        window_type (String): E.g. Gaussian (default). None is a normal window. Accepts any window types listed here:
            https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows
        window_type_std (int): std used for window type
        poisson_cols (list): list with names what should be retrieved from the poisson shot model. Default None (nothing included).
            List can include: "distance", "median_force", "lambda", "f0", "delta", "L"

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
                poisson_rolled = loewe2012.calc(poisson_df, window=window, overlap=(((window - 1) / window) * 100 + 0.0001)) # add epsilon to round up
                poisson_rolled.columns = poisson_all_cols
                poisson_rolled = poisson_rolled[poisson_cols]
            except KeyError:
                print("You can only use a (sub)list of the following features for poisson_cols: distance, median_force, lambda, f0, delta, L")
            # add the poisson data to the all_dfs list and rename columns for distinction
            poisson_rolled.columns = [col + "_" + str(window) for col in poisson_cols]
            all_dfs.append(poisson_rolled)

    return pd.concat(all_dfs, axis=1)

def remove_negatives(df, col="force", threshold=-1):
    """ Remove negative values of a column from dataframe. The values are replaced with the mean of the next and last positive value.
    Parameters:
        df (pd.DataFrame): dataframe
        col (String): column which negative values should get replaced. Default: "force"
        threshold (int): Below which threshold the values should not just get assigned with 0?
    """
    df_removed = df.copy(deep=True)
    # in case the values are just slightly below zero, replace them with 0
    df_removed[(df_removed[col] < 0) & (df_removed[col] > threshold)] = 0
    # if nothing is below a certain threshold, return result
    if not any(df_removed[col] < threshold):
        return df_removed

    # in all other cases replace the negative value with the mean of its next two neighbouring positive values
    index = df_removed.index
    values_neg = df_removed[df_removed[col] < 0]
    indices_neg = index[df_removed[col] < 0]
    # for each negative value
    for value, index in zip(values_neg, indices_neg):
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


def preprocess_profile(profile, target_dir, export_as="csv", sum_mm=1, gradient=False, **kwargs):
    """ Preprocesses a smp profile. Jobs done:
    Indexing, labelling, select data between surface and ground (and relativizes this data).
    Summarizing data in a certain mm window (reduces precision/num of rows).
    Applies a rolling window.
    Exports profile as csv.

    Parameters:
        profile (Profile): the profile which is preprocessed
        target_dir (Path): in which directory the data should be saved
        export_as (String): how the data should be exported. Either as "csv" possible or "npz"
        sum_mm: arg for summarize_rows function - indicates how many mm should be packed together
        gradient (Boolean): arg to decide whether gradient of each datapoint should be calculated
        **kwargs:
            window_size (list): arg for rolling_window function - List of window sizes that should be applied. e.g. [4]
            rolling_cols (list): arg for rolling_window function - List of columns over which should be rolled
            window_type (String): arg for rolling_window function - E.g. Gaussian (default). None is a normal window.
            window_type_std (int): arg for rolling_window function - std used for window type
            poisson_cols (list): arg for rolling_window function - List of features that should be taken from poisson shot model
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
            print("Profile {} is too short for data processing. Profile is skipped".format(profile.name))
            # leave function
            return

    # 1. restrict dataframe between surface and ground (absolute distance values!)
    df = profile.samples_within_snowpack(relativize=False)
    # add label column
    df["label"] = 0

    # 2. label dataframe, if labels are there
    if labelled_data:
        label_pd(df, profile)

    # 3. relativize, such that the first distance value is 0
    relativize(df)

    # 4. Remove all values below 0, replace them with average value around
    if any(df["force"] < 0):
        df_rem = remove_negatives(df)

    # 5. summarize data (precision: 1mm)
    df_mm = summarize_rows(df, mm_window=sum_mm)

    # 6. rolling window in order to know distribution of next and past values (+ poisson shot model)
    final_df = rolling_window(df_mm, **kwargs)

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


    # 8. add cols, index DataFrame and convert dtypes
    final_df["smp_idx"] = idx_to_int(profile.name)

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

def npz_to_pd(npz_file, is_dir):
    """ Converts a npz file to a pandas DataFrame. In case npz_file is a directory,
    all npz files within are loaded, concatenated and returned as one dataframe
    Paramters:
        npz_file (np.npz or Path): A numpy npz file or the path to a npz directory
        is_dir (Boolean): whether npz_file is Path or not
    Returns:
        pd.DataFrame: the converted pandas Dataframe
    """
    if not is_dir:
        smp_npz = np.load(npz_file)
        return pd.DataFrame.from_dict({item: smp_npz[item] for item in smp_npz.files})
    else:
        # match all npz files in the directory
        match_npz = npz_file.as_posix() + "/**/*.npz"
        file_generator = glob.iglob(match_npz, recursive=True)
        # list for all dictionaries
        all_dicts = []
        for file in file_generator:
            # load npz file
            smp_npz = np.load(file)
            # creata dict and save all dicts
            smp_dict = {item: smp_npz[item] for item in smp_npz.files}
            all_dicts.append(smp_dict)
        # merge all dictionaries (columns of first dictionary are used)
        final_dict = {col: np.concatenate([dict[col] for dict in all_dicts]) for col in all_dicts[0]}
        # convert to pandas
        return pd.DataFrame.from_dict(final_dict)

def main():

    print("Starting to export and/or convert data")

    # get temp data
    # tmp = get_temperature(temp=T_LOC)
    # print(tmp.head())

    # export, unite and label smp data
    start = time.time()
    # export data from pnt to csv or npz
    export_pnt(pnt_dir=SMP_LOC, target_dir=EXP_LOC, export_as="npz", overwrite=True, **PARAMS)

    # OTHER OPTIONS
    # unite csv data in one csv file, index it, convert it to pandas (and save it as npz)
    #smp = get_smp_data(csv_dir=EXP_LOC, csv_filename="test04.csv", npz_filename="smp_test04.npz", skip_unify=False, skip_npz=False)

    # FIRST time to use npz_to_pd:
    #smp_first = npz_to_pd(EXP_LOC, is_dir=True)
    # than: export smp as united npz
    #dict = smp_first.to_dict(orient="list")
    #np.savez_compressed("smp_all_final.npz", **dict)

    # AFTER FIRST time and during first time:
    # load pd directly from this npz
    #smp = npz_to_pd("smp_all_final.npz", is_dir=False)

    end = time.time()
    print("Elapsed time for export and dataframe creation: ", end-start)

    print(smp.head())

    print("Number of files in export folder: ", len(os.listdir(EXP_LOC)))
    #print("All pnt files from source dir were also found in the given dataframe: ", check_export(SMP_LOC, smp))

    print_test_df(smp)
    print("Finished export, transformation and printing example features of data.")
# TODO remve smp folders
# TODO: structure this more nicely. remove methods (other file!) which are not useful anymore
# TODO: make it possible to call a method from here in order to get pandas dataframe! (feeds in the constants from above, constants stay default)
# Longterm TODO: make this user friendly - use this with commandline

if __name__ == "__main__":
    main()
