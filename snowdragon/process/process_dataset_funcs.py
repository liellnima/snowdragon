import pandas as pd 

from snowdragon.utils.helper_funcs import normalize

# works only for the original mosaic dataset 
# adapt this function for your own needs
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

# this function removes meltprofiles, renames df particles to pp, and sums up certain classes to be classified as "rare"
def mosaic_specific_processing(smp: pd.DataFrame, labels: dict):
    """ This is implementing specific processing steps for the mosaic data. 
    Implement your own function instead of this one for another dataset. 
    Here, the meltform profiles are excluded. Decomposed/Fragemented particles are renamed to 
    precipation particles (similar enough). The classes if and sh are summed up to the class "rare".
    Parameters:
        smp (pd.DataFrame): smp data 
        labels (dict): label dictionary (str name to int idx)
    Returns:
        pd.DataFrame
    """
    # check: which profiles (and how many) have melted datapoints?
    meltform_profiles = smp.loc[(smp["label"] == labels["mfcl"]) | (smp["label"] == labels["mfcr"]), "smp_idx"].unique()
    #print(smp.loc[(smp["label"] == LABELS["sh"]), "smp_idx"].unique())
    #meltform_profiles_str = [int_to_idx(profile) for profile in meltform_profiles]
    # exclude these profiles!
    smp = smp[~smp["smp_idx"].isin(meltform_profiles)]
    # rename all df points to pp
    smp.loc[smp["label"] == labels["df"], "label"] = labels["pp"]
    # keep: 6, 3, 4, 12, 5, 16, 8, 10: rgwp, dh, dhid, dhwp, mfdh, pp(, if, sh)
    smp = sum_up_labels(
        smp, 
        labels_to_be_unified = ["if", "sh"], 
        unified_label_idx=labels["rare"], 
        all_labels_dict=labels,
        )

    #print(smp["label"].value_counts())
    return smp

def sum_up_labels(
        smp: pd.DataFrame, 
        labels_to_be_unified: list, 
        unified_label_idx: int, 
        all_labels_dict: dict,
    ):
    """ Sums up the datapoints belonging to one of the classes in labels to one class.
    ATTENTION: The new label must be manually added to the graintypes.yaml and the desired
    color to colors.yaml.

    Parameters:
        smp (pd.DataFrame): a dataframe with the smp profiles
        labels_to_be_unified (list): a list of Strings with the labels which should be united
        unified_label_idx (int): number (label index) of the unified label (must be a new number)
        all_labels_dict (dict): dictionary containing all labels, including the new one
    Returns:
        pd.DataFrame: the updated smp
    """
    int_labels = [all_labels_dict[label] for label in labels_to_be_unified]
    smp.loc[smp["label"].isin(int_labels), "label"] = unified_label_idx
    return smp



# should work for all datasets
def normalize_dataset(smp: pd.DataFrame):
    """ Normalizes all the features that should be normalized in the data.
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