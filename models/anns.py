from sklearn.model_selection import train_test_split
from models.metrics import balanced_accuracy, recall, precision, roc_auc_score, log_loss

import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tabulate import tabulate
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, TimeDistributed, Masking

def lstm_architecture(input_shape, output_shape, rnn_size, dropout, dense_units=100, learning_rate=0.01):
    """ The architecture of a lstm model. (Dense Layer, LSTM Layer, Dense Output Layer)
    Parameters:
        input_shape (tuple): contains part of the input shape of the data, namely the maximal length of time points and the number of features
        output_shape (int): contains part of the output shape of the data, namely the number of labels
        rnn_size (int): how many units the LSTM layer should have
        dropout (float): how many percent of the units in the layers should drop out
        dense_units (int): how many units the first dense layer should have. Default=100.
        learning_rate (float): which learning rate the Adam optimizer should use. Default=0.01.
    Return:
        model: a compiled model ready for fitting
    """
    # architecture
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(Dense(units=dense_units, activation="relu"))
    model.add(Dropout(dropout))
    model.add(LSTM(rnn_size, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_shape, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    return model


def blstm_architecture(input_shape, output_shape, rnn_size, dropout, dense_units=100, learning_rate=0.01):
    """ The architecture of a bidirectional lstm model. (Dense Layer, forward LSTM Layer, backward LSTM Layer, Dense Output Layer)
    Parameters:
        input_shape (tuple): contains part of the input shape of the data, namely the maximal length of time points and the number of features
        output_shape (int): contains part of the output shape of the data, namely the number of labels
        rnn_size (int): how many units the LSTM layers should have
        dropout (float): how many percent of the units in the layers should drop out
        dense_units (int): how many units the first dense layer should have. Default=100.
        learning_rate (float): which learning rate the Adam optimizer should use. Default=0.01.
    Return:
        model: a compiled model ready for fitting
    """
    # architecture
    # forward LSTM
    model_forward = Sequential()
    model_forward.add(Masking(mask_value=0.0, input_shape=input_shape))
    model_forward.add(Dense(units=dense_units, activation="relu"))
    model_forward.add(Dropout(dropout))
    model_forward.add(LSTM(rnn_size, return_sequences=True))
    model_forward.add(Dropout(dropout))
    # backward LSTM
    model_back = Sequential()
    model_back.add(Masking(mask_value=0.0, input_shape=input_shape))
    model_back.add(LSTM(rnn_size, return_sequences=True, go_backwards=True))
    model_back.add(Dropout(dropout))
    # bidirectional LSTM
    model = Sequential()
    model.add(Merge([model_forward, model_back], mode="concat"))
    model.add(Dense(units=output_shape, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    return model


def remove_padding(data, profile_len, argmax=True):
    """ Removes the padding of the given data, calculates which label was predicted if wished and flats everything.
    Parameters:
        data (3d np.array): Contains an array with the shape (smp_time_series, smp_data_points, labels)
        profile_len (list): List containing the lengths of all smp profiles from data
        argmax(bool): indicates if the predicted label should be found (argmax=True)
            or the probabilities should be returned (argmax=False). The latter is necessary for probability based scores.
    Returns:
        list: 1d list where all predicted labels (argmax index from one hot encoded labels) are concatenated
    """
    # TODO find out how the probability part works
    # find the predicted label
    argmax_pred = np.argmax(data, axis=2)
    all_preds = []
    for smp, smp_len in zip(argmax_pred, profile_len):
        wanted_smp = smp[:smp_len]
        all_preds.append(wanted_smp)

    # flatten the list
    return [data_point for smp in all_preds for data_point in smp]

# TODO return real values
# TODO add name of the values
# TODO calculate the metrics
# TODO make this more general: eval_ann with ann_type = ["lstm", "blstm", "autoenc"]
def eval_lstm(x_train, x_valid, y_train, y_valid, profile_len_train, profile_len_valid,
             batch_size, epochs, learning_rate, bidirectional, rnn_size, dropout, dense_units, plot_loss=True, file_name="lstm"):
    """ For experimenting purposes.
    Best parameters found so far:
    Learning Rate: 0.01 (smaller -> more epochs)
    Batch Size: 1 (for training 32 might be okay anyway - more epochs in this case)
    Dense Units for dense_lstm or dense_relu_lstm: 100
    Architecture: First a Relu Dense Layer, then LSTM, then maybe bidirectional LSTM, then a last dense layer
    Dropout: unclear
    RNN size: unclear

    Parameters:
        plot_loss (bool): Indicates if plot should be plotted or not
        file_name (str): Name for the plot_loss plot ("_loss") will be added
    """
    # make it tensor data and batch it
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)

    input_shape = (x_train.shape[1], x_train.shape[2])  # (tp_len, features_len)
    output_shape = y_train.shape[-1] # labels_len

    if bidirectional:
        model = blstm_architecture(input_shape=input_shape, output_shape=output_shape, rnn_size=rnn_size, dropout=dropout)
    else:
        model = lstm_architecture(input_shape=input_shape, output_shape=output_shape, rnn_size=rnn_size, dropout=dropout)

    # fitting the model
    start_time = time.time()
    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, workers=-1, verbose=2)
    fit_time = time.time() - start_time

    # plot loss
    if plot_loss:
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        model_type = "BLSTM" if bidirectional else "LSTM"
        plt.title("Model Loss - {}, Batch Size {}, Dropout {}, Learning Rate {}, \nLSTM Size {}, Dense Layer Size {}".format(str(model_type), batch_size, dropout, learning_rate, rnn_size, dense_units))
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend(["train", "validation"], loc='upper left')
        plt.show()
        # if you save the plot:
        if save_plot:
            file_name = file_name + "_loss"
            plt.save_fig(file_name)
            plt.close(fig)

    # predicting
    train_pred = model.predict(train_dataset)
    start_time = time.time()
    valid_pred = model.predict(valid_dataset)
    score_time = time.time() - start_time()

    # remove the paddings
    y_pred_train = remove_padding(train_pred, profile_len_train)
    y_pred_valid = remove_padding(valid_pred, profile_len_valid)
    y_true_train = remove_padding(y_train, profile_len_train)
    y_true_valid = remove_padding(y_valid, profile_len_valid)

    # find out how y_true and y_pred look like at this point

    # calculate the metrics
    scores = {"train_balanced_accuracy": balanced_accuracy(y_true=y_true_train, y_pred=y_pred_train),
              "test_balanced_accuracy": balanced_accuracy(y_true=y_true_valid, y_pred=y_pred_valid),
              "train_recall": recall(y_true=y_true_train, y_pred=y_pred_train),
              "test_recall": recall(y_true=y_true_valid, y_pred=y_pred_valid),
              "train_precision": precision(y_true=y_true_train, y_pred=y_pred_train),
              "test_precision": precision(y_true=y_true_valid, y_pred=y_pred_valid),
              "fit_time": fit_time,
              "score_time": score_time}

    # return the resulting metrics
    return scores


def prepare_data(x, y, smp_idx_all):
    """ One-hot encodes the target data, turns data into numpy arrays of correct shape. Padds the input data such that all time series have the same length.
    Parameters:
        x (pd.DataFrame): feature data
        y (pd.Series): label/target data
        smp_idx (pd.Series): the smp indices for the x and y data
    Returns:
        triple: x_np (np array), y_np (np array), profile_len (list).
            x_np has the shape (time_series, time_points, features) e.g. [(124, 794, 24)].
            y_np has the shape (time_series, time_points, labels) e.g. [(124, 794, 7)].
            profile_len contains (time_series) many entries indicating the length of each smp profile used in x and y
    """
    # one-hot encode target data (3 4 5 6 12 16 17)
    y_one_hot_enc = pd.get_dummies(y)

    # preparation for padding the data
    smp_indices = smp_idx_all.unique()
    # maximal length of a smp time series
    max_smp_len = 0
    for smp in smp_indices:
        if len(x[smp_idx_all == smp]) > max_smp_len:
            max_smp_len = len(x[smp_idx_all == smp])

    # define important and reoccuring dimensions
    ts_len = len(smp_indices) # how many time series we have
    tp_len = max_smp_len # how long the time series maximally are
    features_len = len(x.columns) # how many features we have
    labels_len = len(y_one_hot_enc.columns) # how many labels we have

    # Padding input data (time_series, time_points, features) - to be filled in
    x_np = np.zeros(shape=(ts_len, tp_len, features_len))
    # Padding target data (time_series, time_points, labels) - to be filled in
    y_np = np.zeros(shape=(ts_len, tp_len, labels_len))
    # Saving the original length of each smp profile
    profile_len = []

    # filling in the zeros with data, where data is available
    for smp, smp_idx in zip(smp_indices, range(len(smp_indices))):
        curr_smp = x[smp_idx_all == smp]
        profile_len.append(len(curr_smp))
        for row_index, (row_i, row) in enumerate(curr_smp.iterrows()):
            x_np[smp_idx, row_index, :] = row
            y_np[smp_idx, row_index, :] = y_one_hot_enc.loc[row_i, :] # ATTENTION row_i could be wrong when doing cv!

    # returning the processed data
    return x_np, y_np, profile_len

# TODO make this more general: possible for all architectures
# TODO return tuning results
def tune_lstm(x_train, x_valid, y_train, y_valid, profile_len_train, profile_len_valid, **params):
    """ Tune the LSTM.
    **params:
    batch_size, epochs, learning_rate, bidirectional, rnn_size, dropout, dense_units
    """
    # TODO catch these values from params
    types = ["lstm", "blstm"]
    dropouts = [0, 0.2, 0.5]
    rnn_sizes = [25, 50, 100]
    batch_sizes = [32, 6, 1] # later 1, for velocity reasons 18 at the moment

    # already fixed
    learning_rates = [0.01]
    dense_sizes = [100] # the more the better
    epochs = [50]
    i = 0

    # TODO unpack param dictionary
    for batch_size in batch_sizes:
        for epoch in epochs:
            for learning_rate in learning_rates:
                for dense_units in dense_sizes:
                    for dropout in dropouts:
                        for rnn_size in rnn_sizes:
                            for type in types:
                                bidirectional = True if type=="bidirectional" else False
                                print(bidirectional)
                                print("Running model with the following specs:")
                                print("Model Type {}, Batch size {}, Dropout {}, Learning Rate {}, RNN Size {}, No Dense Units, Epochs {}, Loss Plot {}: \n".format(type, batch_size, dropout, learning_rate, rnn_size, epoch, i))
                                # TODO add epochs!
                                model_results = eval_lstm(profile_len_train, profile_len_valid, x_train, y_train, x_valid, y_valid, i, type, batch_size, learning_rate, rnn_size)

                                result_list = [i, type, batch_size, dropout, learning_rate, rnn_size, dense_units,
                                               model_results["train_balanced_accuracy"], model_results["test_balanced_accuracy"]]

                                # write the results continously in a csv
                                # TODO find out if this works now as intended
                                with open("plots/tables/lstm02.csv", "a+") as text_file:
                                    row_content = ",".join([str(x) for x in result_list])
                                    text_file.write(row_content+'\n')

                                print("Train Bal Acc: ", balanced_accuracy(y_true=y_true_train, y_pred=y_pred_train))
                                print("Valid Bal Acc: ", balanced_accuracy(y_true=y_true_valid, y_pred=y_pred_valid))
                                # increment i
                                i = i + 1

# TODO separate lstm and blstm more clearly
def lstm(x_train, y_train, smp_idx_train, name="LSTM", tuning=True, visualize=False, **params):
    """ LSTM model performing a sequence labeling task. Crossvalidation must respect the time series data!
    """
    # TODO rewrite this for crossvalidation
    # prepare the data
    x_data, y_data, profile_len = prepare_data(x_train, y_train, smp_idx_train)

    # get training and validation data from that
    x_train, x_valid, y_train, y_valid, profile_len_train, profile_len_valid = train_test_split(x_data, y_data, profile_len, test_size=0.2, random_state=42)

    # in the case of tuning
    if tuning:
        tune_lstm(x_train, x_valid, y_train, y_valid, profile_len_train, profile_len_valid, **params)
        # TODO return tuning results
        # TODO save tuning results or return them

    # in all other cases we already know which model we would like to use
    # make crossvalidation for this model
    # return the results

    # make a testing method possible
