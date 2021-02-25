

import pandas as pd
import numpy as np
#import tensorflow as tf


def lstm(x_train, y_train, smp_idx_train, cv, name="LSTM", visualize=False):
    """ LSTM model performing a sequence labeling task. Crossvalidation must respect the time series data!
    """
    # for the moment just first split
    first_fold = cv[0]
    x = x_train.loc[first_fold[0], :]
    x_valid = x_train.loc[first_fold[1], :]
    y = y_train.loc[first_fold[0]]
    y_valid = y_train.loc[first_fold[1]]
    idx_train = smp_idx_train.loc[first_fold[0]]

    # shaping the data correctly
    # input data (time_series, time_points, features)
    smp_indices = idx_train.unique()

    # maximal length of a smp time series
    max_smp_len = 0
    for smp in smp_indices:
        if len(x[idx_train == smp]) > max_smp_len:
            max_smp_len = len(x[idx_train == smp])
    # input data (time_series, time_points, features) - already padded
    x_data = np.zeros(shape=(len(smp_indices), max_smp_len, len(x.columns)))

    for smp, smp_idx in zip(smp_indices, range(len(smp_indices))):
        curr_smp = x[idx_train == smp]
        for row_index, (_, row) in enumerate(curr_smp.iterrows()):
            x_data[smp_idx, row_index, :] = row

    print(x_data.shape)
    exit(0)
    # output data -> one hot encode
    

    # use either batches of size 1 or padd all the other time series

    # parameters

    # architecture
    model = Sequential()
    model.add(Embedding())
    model.add(LSTM())
    model.add(Dropout())
    model.add(LSTM())
    model.add(Dropout())
    model.add(TimeDistributed(Dense(activation="softmax")))

    # fitting the model
