from models.run_models import run_single_model
from models.helper_funcs import load_results

import time
import argparse
import pandas as pd

from pathlib import Path
from csv import DictWriter

# num_cluster = [15, 30], find_num_clusters = ["acc", "bic"], with and without tsne
KMEANS_PARAMS={"num_clusters": 15 , "find_num_clusters": "acc"}

# num_components = [15, 30], cov_type = ["tied, "diag"], find_num_clusters = ["acc", "bic"], with and without tsne
GMM_PARAMS={"num_components": 15, "find_num_clusters": "acc",
            "cov_type": "tied"}

# num_components = [15, 30], cov_type = ["tied, "diag"], find_num_clusters = ["acc", "bic"], with and without tsne
BMM_PARAMS={"num_components": 15, "cov_type": "tied"}

# kernel = ["knn", "rbf"], alpha=[0, 0.2, 0.4]
LABEL_SPREADING_PARAMS={"kernel": "knn", "alpha": 0.2}

# no tuning!!! Only run on subset of complete data with a base model!
SELF_TRAINER_PARAMS={"criterion": "threshold", "base_model": None} # ATTENTION: create base_estimator for this one!

# n_estimators= [25, 100, 500], criterion=["entropy", "gini"]
# max_features=["sqrt", "log2"], max_samples=[0.4, 0.6, 0.8], resample=[False, True]
RF_PARAMS={"n_estimators": 100, "criterion": "entropy", "max_features": "sqrt",
           "max_samples": 0.6, "resample": False}

# decision_function_shape=["ovo", "ovr"], gamma=["auto", "scale"], kernel=["rbf", "sigmoid"]
SVM_PARAMS={"decision_function_shape": "ovr", "gamma": "auto", "kernel": "rbf"}

# n_neighbors=[10, 20, 50, 100, 1000]
KNN_PARAMS={"n_neighbors": 20} # weights should be always distance

# n_estimators=[10, 100, 500], sampling_strategy=["all", "not minority"]
EASY_ENSEMBLE_PARAMS={"n_estimators": 100, "sampling_strategy": "not minority"}

# batch_size=[1, 6, 32], epochs=[50, 100, 150], learning_rate=[0.01, 0.001], dropout=[0, 0.2]
LSTM_PARAMS={"batch_size": 32, "epochs": 15, "learning_rate": 0.01,
             "rnn_size": 100, "dense_units": 100, "dropout": 0.2}

# batch_size=[1, 6, 32], epochs=[50, 100, 150], learning_rate=[0.01, 0.001], dropout=[0, 0.2]
BLSTM_PARAMS={"batch_size": 32, "epochs": 15, "learning_rate": 0.01,
              "rnn_size": 100, "dense_units": 100, "dropout": 0.2}

# batch_size=[1, 6, 32], epochs=[50, 100, 150], learning_rate=[0.001, 0.0001], dropout=[0, 0.2],
# attention=[False, True], bidirectional=[True, False]
ENC_DEC_PARAMS={"batch_size": 32, "epochs": 15, "learning_rate": 0.001,
                "rnn_size": 100, "dense_units": 0, "dropout": 0.2,
                "attention": False, "bidirectional": False, "regularizer": False}

parser = argparse.ArgumentParser(description="Can be used for tuning, runs a single model. Parameters are specified arguments. All arguments have default parameters from the tuning_parameters.py file.")

# Mandatory arguments

# hyperparameters

def main():
    """ Read in Parameters and tune a model with these parameters.
    """
    #args = parser.parse_args()
    #params = vars(args)
    #print("Parameters used are:\n", params)

    data = load_results("preprocessed_data_dict.txt")
    # single file for each model needed, since they have different parameters
    save_in = "plots/tables/" + "tuning_results_test.csv"
    params = ENC_DEC_PARAMS
    #tsne_data = load_results("preprocessed_tsne_dict.txt")

    # Running the model
    scores = run_single_model(model_type="enc_dec", data=data, **params)
    scores_and_params = {**params, **scores}
    print(scores_and_params)

    # Saving the results if wished
    if save_in is not None:
        file_exists = Path(save_in).is_file()
        with open(save_in, 'a+') as csvfile:
            fieldnames = list(scores_and_params.keys())
            writer = DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(scores_and_params)
        # TODO delete this, just for testing purposes here
        results = pd.read_csv(save_in)
        print(results)

    # TUNING:
    # run each model with different bash scripts
    #

if __name__ == "__main__":
    main()
