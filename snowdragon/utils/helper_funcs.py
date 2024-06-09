import re
import glob
import yaml
import pickle
import numpy as np
import pandas as pd

from snowdragon import CONFIG_DIR

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
    elif "S36M" in string_idx:
        return int("5" + string_idx[-4:].zfill(6))
    else:
        raise ValueError("SMP naming convention is unknown. Please add another elif line in idx_to_int to handle your SMP naming convention.")


def int_to_idx(int_idx: int) -> str:
    """ Converts an int that indexes the smp profile to a string.
    Paramters:
        int_idx (int): the index that is converted
    Returns:
        str: the index as string
        For smp profiles starting with 200, 300 or 400, the last four digits are caught
            and added either to       S31H, S43M, S49M.
        For smp profiles starting with 1 throw an error, since no SMP profile should have a 1.
            PS122 is only used to describe event ids.
        Profiles with 0 throw an error.
        All other profiles get their int index returned as string (unchanged).
    """
    int_idx = str(int_idx)
    smp_device = int(int_idx[0])
    if (smp_device == 1) or (smp_device == 0):
        raise ValueError("SMP indices with 0 or 1 cannot be converted. Indices with 1 are reserved for event IDs. 0 means that no suitable match was found during index convertion.")
    elif smp_device == 2:
        return "S31H" + int_idx[3:7]
    elif smp_device == 3:
        return "S43M" + int_idx[3:7]
    elif smp_device == 4:
        return "S49M" + int_idx[3:7]
    elif smp_device == 5:
        return "S36M" + int_idx[3:7]
    else:
        return str(int_idx)

def normalize(data: pd.DataFrame, features, min: int, max: int) -> pd.DataFrame:
    """ Normalizes the given features of a dataframe.
    Parameters:
        data (pd.DataFrame): the dataframe to normalize
        features (list or str): list of strings or a single string indicating the feature to normalize
            If several feature they must share the same min and max
        min (int): the minimum of the features
        max (int): the maximum of the features
    Returns:
        pd.DataFrame: data with normalized features
    """
    data.loc[:, features] = data.loc[:, features].apply(lambda x: (x - min) / (max - min))
    return data

def reverse_normalize(data: pd.DataFrame, features, min: int, max: int) -> pd.DataFrame:
    """ Reverses the normalization of the given features of a dataframe.
    Parameters:
        data (pd.DataFrame): the dataframe to reverse normalize
        features (list or str): list of strings or a single string indicating the feature to reverse normalize.
            If several feature they must share the same min and max.
        min (int): the minimum of the features used during normalization
        max (int): the maximum of the features used during normalization
    Returns:
        pd.DataFrame: data with reversed features
    """
    data.loc[:, features] = data.loc[:, features].apply(lambda x: (x * (max - min)) + min)
    return data

def save_results(file_name: str, object):
    """ Pickels an object.
    Parameters:
        file_name (str): under which filename the object should be saved
        object (obj): some python object that can be pickled
    """
    with open(file_name, "wb") as myFile:
        pickle.dump(object, myFile)

def load_results(file_name: str):
    """ Loads the data from a pickle file.
    Parameters:
        file_name (str): under which name the data was saved
    Returns:
        (obj): the data object
    """
    with open(file_name, "rb") as myFile:
        data = pickle.load(myFile)
    return data

def load_configs(config_subdir: str, config_name: str) -> dict:
    """ Loads the configs from a yaml file. 
    Parameters:
        config_subdir (str): In which subdir the configs are stored
        config_name (str): The name of the configs 
    Returns:
        dict: The configs in form of a dictionary
    """
    with open(CONFIG_DIR / config_subdir / config_name) as file:
        try:
            configs = yaml.safe_load(file)
        except yaml.YAMLErrot as err:
            print(err)

    return configs

def load_smp_data(npz_name, test_print=False, **kwargs):
    """ Wrapper Function for npz_to_pd for easier usage. Returns the data from a npz file as pd.DataFrame.
    Paramters:
        npz_file (String): Name of the npz file to load
        test (Boolean): Default false. Indicates if some information should be printed out about the data. Can be used for testing purposes.
    Returns:
        pd.DataFrame: the data from the npz file loaded into a dataframe
    """
    if test_print:
        smp = npz_to_pd(npz_name, is_dir=False)
        print_test_df(smp)
        return smp
    else:
        return npz_to_pd(npz_name, is_dir=False)

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

