from sklearn.model_selection import train_test_split
from models.metrics import balanced_accuracy, recall, precision, roc_auc, my_log_loss, calculate_metrics_raw, METRICS, METRICS_PROB

import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from tabulate import tabulate
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Masking, Bidirectional, Activation, Input, RepeatVector
from keras_self_attention import SeqSelfAttention


def lstm_architecture(input_shape, output_shape, rnn_size=100, dropout=0, dense_units=100, learning_rate=0.01, **kwargs):
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
    if dense_units > 0:
        model.add(Dense(units=dense_units, activation="relu"))
        model.add(Dropout(dropout))
    model.add(LSTM(rnn_size, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_shape, activation="relu"))
    model.add(Activation("softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    return model


def blstm_architecture(input_shape, output_shape, rnn_size=100, dropout=0, dense_units=100, learning_rate=0.01, **kwargs):
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
    # change this: https://stackoverflow.com/questions/62991082/bidirectional-lstm-merge-mode-explanation
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    if dense_units > 0:
        model.add(Dense(units=dense_units, activation="relu"))
        model.add(Dropout(dropout))
    forward_layer = LSTM(rnn_size, return_sequences=True)
    backward_layer = LSTM(rnn_size, return_sequences=True, go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode="concat"))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_shape, activation="relu"))
    model.add(Activation("softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    return model

# package I use for attention: https://github.com/CyberZHG/keras-self-attention
# https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#whats-wrong-with-seq2seq-model
# https://www.tensorflow.org/tutorials/text/nmt_with_attention
# https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
def enc_dec_architecture(input_shape, output_shape, rnn_size=100, dropout=0.2, dense_units=0, learning_rate=0.001, attention=False, bidirectional=False, regularize=True, **kwargs):
    """ An Encoder-Decoder Architecture - if wished with attention mechanism.
    Parameters:
        input_shape (tuple): contains part of the input shape of the data, namely the maximal length of time points and the number of features
        output_shape (int): contains part of the output shape of the data, namely the number of labels
        rnn_size (int): how many units the LSTM layers should have
        dropout (float): how many percent of the units in the layers should drop out
        dense_units (int): how many units the first dense layer should have. Default=0 means that no feed forward network is used in the beginning.
        learning_rate (float): which learning rate the Adam optimizer should use.
            Default=0.001, since the encoder-decoder needs a smaller learning rate than simpler models.
        bidirectional (bool): if the encoder should be bidirectional. Default=False.
        regularize (bool): sets a regularizer for the encoder and decoder. Default=True means that the regularizer is set.
    """
    lstm_regularizers = regularizers.l1_l2(l1=0.001, l2=0.001) if regularize else None
    enc_return_seq = True if attention else False

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    # TODO a boolean to decide if there should be stacked (B)LSTMs instead of a dense layer
    # first dense layer
    if dense_units > 0:
        model.add(Dense(units=dense_units, activation="relu"))
        model.add(Dropout(dropout))

    # encoder
    if bidirectional:
        forward_layer = LSTM(rnn_size, return_sequences=enc_return_seq)
        backward_layer = LSTM(rnn_size, return_sequences=enc_return_seq, go_backwards=True)
        model.add(Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode="concat"))
        model.add(Dropout(dropout))
    else:
        model.add(LSTM(rnn_size, kernel_regularizer=lstm_regularizers, return_sequences=enc_return_seq))
        model.add(Dropout(dropout))

    # attention if wished
    if attention:
        # global attention -> can be changed to local attention with attention_width
        model.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD, # attention_type (ADD: Bahadenau) (MUL: Luong)
                                   units=32, # units for feed forward network of attention layer
                                   attention_activation="tanh")) # as in Bahadenau
    # bottleneck
    else:
        model.add(RepeatVector(input_shape[0]))

    # decoder
    model.add(LSTM(rnn_size, kernel_regularizer=lstm_regularizers, return_sequences=True))
    model.add(Dropout(dropout))
    # output layer
    model.add(Dense(units=output_shape, activation="relu"))
    # softmax for probability
    model.add(Activation("softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    return model


def remove_padding(data, profile_len, return_prob=False):
    """ Removes the padding of the given data, calculates which label was predicted if wished and flats everything.
    Parameters:
        data (3d np.array): Contains an array with the shape (smp_time_series, smp_data_points, labels)
        profile_len (list): List containing the lengths of all smp profiles from data
        return_prob(bool): indicates if the predicted label should be found (return_prob=False)
            or the probabilities should be returned (return_prob=True). The latter is necessary for probability based scores.
    Returns:
        list: 1d list where all predicted labels (argmax index from one hot encoded labels) are concatenated
    """
    # find the predicted label if wished
    argmax_pred = np.argmax(data, axis=2) if not return_prob else data

    all_preds = []
    for smp, smp_len in zip(argmax_pred, profile_len):
        wanted_smp = smp[:smp_len]
        all_preds.append(wanted_smp)

    # flatten the list
    return [data_point for smp in all_preds for data_point in smp]

# TODO rename variables (more consistent!)
# TODO documentation
# best params so far:
    # For experimenting purposes.
    # Best parameters found so far:
    # Learning Rate: 0.01 (smaller -> more epochs)
    # Batch Size: 1 (for training 32 might be okay anyway - more epochs in this case)
    # Dense Units for dense_lstm or dense_relu_lstm: 100
    # Architecture: First a Relu Dense Layer, then LSTM, then maybe bidirectional LSTM, then a last dense layer
    # Dropout: unclear
    # RNN size: unclear
def eval_ann(x_train, x_valid, y_train, y_valid, profile_len_train, profile_len_valid,
             ann_type, batch_size=32, epochs=15, plot_loss=False, file_name=None, **params):
    """ This function can evaluate a single ANN. The functions returns the evaluation results and plots the loss if wished.

    Parameters:
        x_train (np.array): the input training data
        x_valid (np.array): the input validation data
        y_train (np.array): the target training data
        y_valid (np.array): the target validation data
        profile_len_train (list): contains all lengths of each profile in the training data
        profile_len_valid (list): contains all lengths of each profile in the validation data
        batch_size (int): batch size employed during ANN training
        epochs (int): number of epochs run during ANN training
        ann_type (str): describing the choosen ann type. Can be on of the following three: ["lstm", "blstm", "enc_dec"]
        plot_loss (bool): Indicates if plot should be plotted or not
        file_name (str): Name for the plot_loss plot ("_loss") will be added. Default = None means that the plot won't be saved.
        **params (dict): The following parameters for the ANN architectures appear here:
            rnn_size, dense_units, dropout, regularizer, attention, bidirectional, learning_rate
    Returns:
        dict: contains y_true_train, y_true_valid, y_pred_train, y_pred_valid, y_pred_prob_train, y_pred_prob_valid
    """
    # make it tensor data and batch it
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)

    input_shape = (x_train.shape[1], x_train.shape[2])  # (tp_len, features_len)
    output_shape = y_train.shape[-1] # labels_len

    if ann_type == "blstm":
        model = blstm_architecture(input_shape=input_shape, output_shape=output_shape, **params)
    elif ann_type == "lstm":
        model = lstm_architecture(input_shape=input_shape, output_shape=output_shape, **params)
    elif ann_type == "enc_dec":
        model = enc_dec_architecture(input_shape=input_shape, output_shape=output_shape, **params)
    else:
        raise ValueError("Parameter \"ann_type\" in eval_ann() must be one of the following: \"lstm\", \"blstm\" or \"enc_dec\".")

    # fitting the model
    start_time = time.time()
    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, workers=-1, verbose=2)
    fit_time = time.time() - start_time

    # plot loss
    if plot_loss:
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model Loss - {}, Batch Size {}, Dropout {}, Learning Rate {}, \nLSTM Size {}, Dense Layer Size {}".format(ann_type, batch_size, dropout, learning_rate, rnn_size, dense_units))
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend(["train", "validation"], loc='upper left')
        plt.show()
        # if you save the plot:
        if file_name is not None:
            file_name = file_name + "_loss"
            plt.save_fig(file_name)
            plt.close(fig)

    # predicting
    train_pred = model.predict(train_dataset)
    start_time = time.time()
    valid_pred = model.predict(valid_dataset)
    score_time = time.time() - start_time

    # remove the paddings and separate between probability predictions and direct predictions
    y_pred_train = remove_padding(train_pred, profile_len_train)
    y_pred_valid = remove_padding(valid_pred, profile_len_valid)
    y_pred_prob_train = remove_padding(train_pred, profile_len_train, return_prob=True)
    y_pred_prob_valid = remove_padding(valid_pred, profile_len_valid, return_prob=True)
    y_true_train = remove_padding(y_train, profile_len_train)
    y_true_valid = remove_padding(y_valid, profile_len_valid)

    results = {"fit_time": fit_time, "score_time": score_time, "y_true_train": y_true_train, "y_true_valid": y_true_valid,
               "y_pred_train": y_pred_train, "y_pred_valid": y_pred_valid, "y_pred_prob_train": y_pred_prob_train, "y_pred_prob_valid": y_pred_prob_valid}

    # return the results
    return results


def prepare_data(x, y, smp_idx_all, labels=None, max_smp_len=None):
    """ One-hot encodes the target data, turns data into numpy arrays of correct shape. Padds the input data such that all time series have the same length.
    Parameters:
        x (pd.DataFrame): feature data
        y (pd.Series): label/target data
        smp_idx (pd.Series): the smp indices for the x and y data
        labels (list): Labels occurring in x and y. If None (default) this value is inferred from the target data.
        max_smp_len (int): the maximal profile length (used for padding) in the dataset.
            Default=None means that this will be inferred from the given data.
    Returns:
        triple: x_np (np array), y_np (np array), profile_len (list).
            x_np has the shape (time_series, time_points, features) e.g. [(124, 794, 24)].
            y_np has the shape (time_series, time_points, labels) e.g. [(124, 794, 7)].
            profile_len contains (time_series) many entries indicating the length of each smp profile used in x and y
    """
    # usual case: one hot encode the target data
    if labels is None:
        # one-hot encode target data (3 4 5 6 12 16 17)
        y_one_hot_enc = pd.get_dummies(y)
    # if one of the labels is missing: append it to y and delete last row again
    else:
        if len(y.unique()) == len(labels):
            y_one_hot_enc = pd.get_dummies(y)
        else:
            missing_labels = set(labels) - set(y.unique())
            y_one_hot_enc = pd.get_dummies(y)
            # append the missing labels as columns
            for missing_label in missing_labels:
                y_one_hot_enc[missing_label] = 0
            # sort columns
            y_one_hot_enc = y_one_hot_enc.sort_index(axis=1)

    # preparation for padding the data
    smp_indices = smp_idx_all.unique()
    # maximal length of a smp time series
    if max_smp_len is None:
        max_smp_len = 0
        for smp in smp_indices:
            if len(x[smp_idx_all == smp]) > max_smp_len:
                max_smp_len = len(x[smp_idx_all == smp])

    # define important and reoccuring dimensions
    ts_len = len(smp_indices) # how many time series we have
    tp_len = max_smp_len # how long the time series maximally are
    features_len = len(x.columns) # how many features we have
    labels_len = len(y_one_hot_enc.columns) if labels is None else len(labels) # how many labels we have

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
            y_np[smp_idx, row_index, :] = y_one_hot_enc.loc[row_i, :]

    # returning the processed data
    return x_np, y_np, profile_len

# TODO make this more general: possible for all architectures
# TODO return tuning results
# TODO documentation
def tune_lstm(x_train, x_valid, y_train, y_valid, profile_len_train, profile_len_valid, **params):
    """ Tune the LSTM.
    **params:
    batch_size, epochs, learning_rate, ann_type, rnn_size, dropout, dense_units, attention, bidirectional, regularizer
    """
    # TODO catch these values from params
    ann_types = ["lstm", "blstm", "enc_dec"]
    dropouts = [0, 0.2, 0.5]
    rnn_sizes = [25, 50, 100]
    batch_sizes = [32, 6, 1] # later 1, for velocity reasons 18 at the moment

    # already fixed
    learning_rates = [0.01]
    dense_sizes = [100] # the more the better
    epochs_sizes = [50]
    i = 0

    # TODO unpack param dictionary
    for batch_size in batch_sizes:
        for epochs in epochs_sizes:
            for learning_rate in learning_rates:
                for dense_units in dense_sizes:
                    for dropout in dropouts:
                        for rnn_size in rnn_sizes:
                            for ann_type in ann_types:
                                print("Running model with the following specs:")
                                print("Model Type {}, Batch size {}, Dropout {}, Learning Rate {}, RNN Size {}, No Dense Units, Epochs {}, Loss Plot {}: \n".format(ann_type, batch_size, dropout, learning_rate, rnn_size, epochs, i))
                                model_results = eval_ann(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid,
                                                          profile_len_train=profile_len_train, profile_len_valid=profile_len_valid,
                                                          batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
                                                          ann_type=ann_type, rnn_size=rnn_size, dropout=dropout, dense_units=dense_units)

                                result_list = [i, ann_type, batch_size, dropout, learning_rate, rnn_size, dense_units,
                                               model_results["train_balanced_accuracy"], model_results["test_balanced_accuracy"]]
                                exit(0)
                                # write the results continously in a csv
                                # TODO find out if this works now as intended
                                with open("plots/tables/lstm_test.csv", "a+") as text_file:
                                    row_content = ",".join([str(x) for x in result_list])
                                    text_file.write(row_content+'\n')

                                print("Train Bal Acc: ", balanced_accuracy(y_true=y_true_train, y_pred=y_pred_train))
                                print("Valid Bal Acc: ", balanced_accuracy(y_true=y_true_valid, y_pred=y_pred_valid))
                                # increment i
                                i = i + 1
# TODO make this somewhere outside
def print_tuning(file_name):
    """ Read out csv file with tuning results and print some information.
    Parameters:
        file_name (str): csv file name with tuning results
    """
    results = pd.read_csv(file_name)
    results = results.dropna(axis=1, how="all")

    print("10 Best Models regarding Validation Accuracy:\n")
    print(results.sort_values(by="BalAccValid", ascending=False).iloc[0:10, :])

    parameters = results.columns[0:-2]

    for param in parameters:
        print("Mean Accuracy results per value for parameter {}".format(param))
        print(results.groupby(param)["BalAccValid"].mean())
        print(results.groupby(param)["BalAccTrain"].mean())
        print("Max Accuracy results per value for parameter {}".format(param))
        print(results.groupby(param)["BalAccValid"].max())
        print(results.groupby(param)["BalAccTrain"].max())

def calculate_metrics_ann(results, name=None):
    """ Calculates all the Metrics from METRICS and METRICS_PROB for a given dictionary.
    Parameters:
        results (dict): Dictionary with the following keys:
            fit_time, score_time, y_true_train, y_true_valid, y_pred_train, y_pred_valid, y_pred_prob_train, y_pred_prob_valid
        name (str): Default=None means no column for names is added, otherwise set the model name with this.
    Returns:
        dict: with all scores from METRICS and METRICS_PROB for training and validation/testing data
    """
    all_scores = {"fit_time": np.array(results["fit_time"]), "score_time": np.array(results["score_time"])}
    all_scores.update(calculate_metrics_raw(y_trues=results["y_true_train"], y_preds=results["y_pred_train"], metrics=METRICS, cv=False, annot="train"))
    all_scores.update(calculate_metrics_raw(y_trues=results["y_true_valid"], y_preds=results["y_pred_valid"], metrics=METRICS, cv=False, annot="test"))
    try:
        all_scores.update(calculate_metrics_raw(y_trues=results["y_true_train"], y_preds=results["y_pred_prob_train"], metrics=METRICS_PROB, cv=False, annot="train"))
        all_scores.update(calculate_metrics_raw(y_trues=results["y_true_valid"], y_preds=results["y_pred_prob_valid"], metrics=METRICS_PROB, cv=False, annot="test"))
    except ValueError as e:
        # print the non-probability based scores that were possible to calculate
        print(all_scores)
        # and raise a ValueError for the rest
        if str(e) == "Number of classes in y_true not equal to the number of columns in 'y_score'":
            raise ValueError("""The train-test split you chose is most likely not stratified.
                Some of the classes occuring in the training dataset do not occur in the testing dataset.
                Try to fix this or calculate the metrics after crossvalidation.""")
        elif str(e) == "Input contains NaN, infinity or a value too large for dtype('float32').":
            print("Rerun experiments with different parameters, current model had nans as loss.")
            for metric_prob in METRICS_PROB.values():
                all_scores["train_" + metric_prob] = float("nan")
                all_scores["test_" + metric_prob] = float("nan")
        else:
            raise
    if name is not None:
        all_scores["model"] = name

    return all_scores

def find_max_len(smp_idx, x_data):
    """ Finds the maximum length of a profile in this dataset.
    Parameters:
        smp_idx (np.array): The smp indices for each data point.
        x_data (np.array): The input data, the smp profiles.
    """
    max_smp_len = 0
    for smp in smp_idx.unique():
        if len(x_data[smp_idx == smp]) > max_smp_len:
            max_smp_len = len(x_data[smp_idx == smp])

    return max_smp_len

# TODO include RNN
# TODO include ANN
# TODO include print tuning and tuning function somewhere else:
    # read out csv and check results
    # TODO delete this
    #print_tuning("plots/tables/lstm02.csv")
# TODO fix warning with incompatible shape
def ann(x_train, y_train, smp_idx_train, ann_type="lstm", name="LSTM", cv_timeseries=0.2, plot_loss=False, plot_loss_name=None, **params):
    """ The wrapper function to run any ANN architecture provided in this file.
    Parameters:
        x_train (pd.DataFrame): the complete training data
        y_train (pd.DataFrame): the complete target data
        smp_idx_train (pd.Series): smp index for each training sample
        ann_type (str): describing the choosen ann type. Can be on of the following three: ["lstm", "blstm", "enc_dec"]
        name (str): how the model should be named
        cv_timeseries (float or list): float indicates that a simple percentual training-test split should be done.
            Otherwise it must be an iteratable list of length k with tuples of np 1-d arrays (train_indices, test_indices).
        plot_loss (bool): if the loss should be printed after training.
        plot_loss_name (str): Name for the plot_loss plot ("_loss") will be added. Default = None means that the plot won't be saved.
        **params (dict): containing: batch_size, epochs, learning_rate, rnn_size, dropout, dense_units,
            bidirectional, attention, regularizer
    Returns:
        dict: all the metrics calculated from the different crossvalidation splits. Each key is connected to list, where the results are stored.
    """
    # get training and validation data from that
    if isinstance(cv_timeseries, float):
        # make the train and validation split
        x_train, x_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(x_train, y_train, smp_idx_train, test_size=cv_timeseries, random_state=42)

        # prepare the data
        max_smp_len_train = find_max_len(smp_idx=idx_train, x_data=x_train)
        max_smp_len_valid = find_max_len(smp_idx=idx_valid, x_data=x_valid)

        x_train, y_train, profile_len_train = prepare_data(x_train, y_train, idx_train, max_smp_len=max_smp_len_train, labels=list(y_train.unique()))
        x_valid, y_valid, profile_len_valid = prepare_data(x_valid, y_valid, idx_valid, max_smp_len=max_smp_len_valid, labels=list(y_valid.unique()))

        # we already know which model we would like to use:
        results = eval_ann(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid,
                                  profile_len_train=profile_len_train, profile_len_valid=profile_len_valid,
                                  ann_type=ann_type, plot_loss=plot_loss, file_name=plot_loss_name, **params)

        # call metrics on model_results
        return calculate_metrics_ann(results, name=name)

    else:
        # all results
        all_results = {} # the rest is added on its own
        keys_assigned = False
        labels = list(y_train.unique())
        # for each fold in the crossvalidation
        for k in tqdm(cv_timeseries):
            # find maximal smp profile length in complete dataset
            #max_smp_len = find_max_len(smp_idx=smp_idx_train, x_data=x_train)
            max_smp_len = 0
            for smp in smp_idx_train.unique():
                if len(x_train[smp_idx_train == smp]) > max_smp_len:
                    max_smp_len = len(x_train[smp_idx_train == smp])
            # mask the data and prepare it
            x_k_train, y_k_train, profile_len_k_train = prepare_data(x_train.iloc[k[0]], y_train.iloc[k[0]], smp_idx_train.iloc[k[0]], max_smp_len=max_smp_len, labels=labels)
            x_k_valid, y_k_valid, profile_len_k_valid = prepare_data(x_train.iloc[k[1]], y_train.iloc[k[1]], smp_idx_train.iloc[k[1]], max_smp_len=max_smp_len, labels=labels)

            # run the models with this cv split
            k_results = eval_ann(x_train=x_k_train, x_valid=x_k_valid, y_train=y_k_train, y_valid=y_k_valid,
                                      profile_len_train=profile_len_k_train, profile_len_valid=profile_len_k_valid,
                                      plot_loss=plot_loss, ann_type=ann_type, file_name=plot_loss_name, **params)
            if not keys_assigned:
                all_results = {k: [] for k in k_results.keys()}
                keys_assigned = True

            # append the results to the dict
            for key in k_results.keys():
                if "time" not in key:
                    all_results[key] = all_results[key] + k_results[key]
                elif "score_time" in key:
                    all_results["score_time"].append(k_results[key])
                elif "fit_time" in key:
                    all_results["fit_time"].append(k_results[key])

        return calculate_metrics_ann(all_results, name=name)

    # TODO make a testing method possible
