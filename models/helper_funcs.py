import pickle
import pandas as pd
import numpy as np


# longterm TODO: move idx_to_int from data_preprocessing here!

def int_to_idx(int_idx):
    """ Converts an int that indexes the smp profile to a string.
    Paramters:
        int_idx (String): the index that is converted
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

def normalize(data, features, min, max):
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

def reverse_normalize(data, features, min, max):
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

def save_results(file_name, object):
    """ Pickels an object.
    Parameters:
        file_name (str): under which filename the object should be saved
        object (obj): some python object that can be pickled
    """
    with open(file_name, "wb") as myFile:
        pickle.dump(object, myFile)

def load_results(file_name):
    """ Loads the data from a pickle file.
    Parameters:
        file_name (str): under which name the data was saved
    Returns:
        (obj): the data object
    """
    with open(file_name, "rb") as myFile:
        data = pickle.load(myFile)
    return data
