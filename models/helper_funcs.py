import pickle
import pandas as pd
import numpy as np

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
