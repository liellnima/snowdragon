from sklearn.model_selection import train_test_split

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
