from models.run_models import run_single_model
from models.helper_funcs import load_results
from tuning.tuning_parameters import KMEANS_PARAMS, GMM_PARAMS, BMM_PARAMS, LABEL_SPREADING_PARAMS
from tuning.tuning_parameters import SELF_TRAINER_PARAMS, RF_PARAMS, SVM_PARAMS, KNN_PARAMS, EASY_ENSEMBLE_PARAMS
from tuning.tuning_parameters import LSTM_PARAMS, BLSTM_PARAMS, ENC_DEC_PARAMS, FIELD_NAMES
from data_handling.data_parameters import SMP_PREPROCESSED_TXT

import argparse
import pandas as pd

from pathlib import Path
from csv import DictWriter


parser = argparse.ArgumentParser(description="Can be used for tuning, runs a single model. Parameters are specified arguments. Only a subset of the arguments are relevant when running one model. All arguments have default parameters from the tuning_parameters.py file.")

# Mandatory arguments
parser.add_argument("output", type=str, help="Name of the csv output file where the results are saved. Must have a .csv ending.")

# model and data arguments
# default was: "data/preprocessed_data_k5_updated02.txt"
parser.add_argument("--data_file", default=SMP_PREPROCESSED_TXT, type=str,
                    help="Name of the file where the preprocessed data is stored.")
parser.add_argument("--model_type", default="baseline", type=str,
                    help="""Must be one of the following models: \"baseline\",
                    \"kmeans\", \"gmm\", \"bmm\", \"label_spreading\", \"self_trainer\", \"rf\", \"svm\",
                    \"knn\", \"easy_ensemble\", \"lstm\", \"blstm\", \"enc_dec\"""")

# name of the model
parser.add_argument("--name", default=None, type=str, help="Name of the model. Will be written in the model category in the results.")

# hyperparameters
parser.add_argument("--num_clusters", default=KMEANS_PARAMS["num_clusters"], type=int,
                    help="How many clusters should be searched for in kmeans clustering.")
parser.add_argument("--find_num_clusters", default="both", type=str,
                    help="Can be either \"acc\", \"sil\", \"bic\" or \"both\". Indicates the strategy how the optimal number of clusters or components should be found.")
parser.add_argument("--num_components", default=BMM_PARAMS["num_components"], type=int,
                    help="Number of maximal components for (Bayesian) Gaussian Mixture Models.")
parser.add_argument("--cov_type", default=GMM_PARAMS["cov_type"], type=str,
                    help="Can be either \"tied\", \"diag\", \"full\" or \"spherical\". Covariance type that is used for (Bayesian) Gaussian Mixture Models.")
parser.add_argument("--kernel", default="rbf", type=str,
                    help="Can be either \"knn\" or \"rbf\" for Kernel used during Label Spreading. Can be either \"linear\", \"poly\", \"rbf\" or \"sigmoid\" for SVM kernel.")
parser.add_argument("--alpha", default=LABEL_SPREADING_PARAMS["alpha"], type=float,
                    help="Clamping factor for label spreading. Must be between 0 and 1. 0 means that no information from neighbours is adopted to.")
parser.add_argument("--criterion", default=RF_PARAMS["criterion"], type=str,
                    help="Attention, parameter can be used for Random Forest and Self Trainer. Default is set to a RF value. For self training the criterion can be \"threshold\" or \"k_best\". For the random forest it can be \"entropy\" or \"gini\".")
parser.add_argument("--n_estimators", default=RF_PARAMS["n_estimators"], type=int,
                    help="Parameter for the number of deciions trees employed in Random Forest or Easy Ensemble.")
parser.add_argument("--max_features", default=RF_PARAMS["max_features"], type=str,
                    help="Can be either \"sqrt\" or \"log2\". Determines how many features should be bagged for the Random Forest.")
parser.add_argument("--max_samples", default=RF_PARAMS["max_samples"], type=float,
                    help="Value between 0 and 1. Indicates how many percent of the data should be used in one decision tree.")
parser.add_argument("--resample", default=int(RF_PARAMS["resample"]), type=int,
                    help="0 for false and 1 for true to indicate if resampling for imbalanced data should be done in the Random Forest.")
parser.add_argument("--decision_function_shape", default=SVM_PARAMS["decision_function_shape"], type=str,
                    help="Can be either \"ovr\" or \"ovo\". Multiclass classification strategy for SVMs.")
parser.add_argument("--gamma", default=SVM_PARAMS["gamma"], type=str,
                    help="Can be either \"auto\" or \"scale\". Kernel coefficient for SVM.")
parser.add_argument("--n_neighbors", default=KNN_PARAMS["n_neighbors"], type=int,
                    help="How many neighbours should be considered in KNN classifier.")
parser.add_argument("--sampling_strategy", default=EASY_ENSEMBLE_PARAMS["sampling_strategy"], type=str,
                    help="Can be either \"not minority\", \"all\", \"not majority\" or \"majority\". This is the sampling strategy for the Easy Ensemble that deals with imbalanced datasets.")
parser.add_argument("--batch_size", default=BLSTM_PARAMS["batch_size"], type=int,
                    help="Batch size for ANNs.")
parser.add_argument("--epochs", default=BLSTM_PARAMS["epochs"], type=int,
                    help="Number of training epochs for ANNs.")
parser.add_argument("--learning_rate", default=BLSTM_PARAMS["learning_rate"], type=float,
                    help="Learning rate for ANNs. Should be small, e.g. 0.001")
parser.add_argument("--rnn_size", default=BLSTM_PARAMS["rnn_size"], type=int,
                    help="How many recurrent units in the recurrent layers should be employed in the ANNs.")
parser.add_argument("--dense_units", default=BLSTM_PARAMS["dense_units"], type=int,
                    help="How many dense units should be used in the feedforward layer before each RNN architecture. Can be set to 0 such that no feedforward layer is employed.")
parser.add_argument("--dropout", default=BLSTM_PARAMS["dropout"], type=float,
                    help="Value between 0 and 1 indicating how many units (in percentage) should be randomly shut off during training to avoid overfitting.")
parser.add_argument("--attention", default=int(ENC_DEC_PARAMS["attention"]), type=int,
                    help="0 for false and 1 for true to indicate if attention mechanism should be used or not in the Encoder-Decoder network.")
parser.add_argument("--bidirectional", default=int(ENC_DEC_PARAMS["bidirectional"]), type=int,
                    help="0 for false and 1 for true to indicate if RNN layers in the Encoder-Decoder network should be bidirectional or not.")
parser.add_argument("--regularize", default=int(ENC_DEC_PARAMS["regularize"]), type=int,
                    help="0 for false and 1 for true to indicate if a fixed l1 and l2 regularizer should be employed")
parser.add_argument("--print_results", default=0, type=int,
                    help="0 for do not print results and 1 for printing results. Does nothing else than printing results.")

def main():
    """ Read in Parameters and tune a model with these parameters.
    """
    args = parser.parse_args()
    params = vars(args)

    # get the data file and the output file
    data_file_name = Path(params["data_file"])
    save_in = params["output"]
    params["sampling_strategy"] = params["sampling_strategy"].replace("_", " ")

    if params["print_results"]:
        results = pd.read_csv(save_in)
        print(results)
        exit(0)

    data = load_results(data_file_name)

    # load potential tsne dimension reduced data
    #tsne_data = load_results("data/preprocessed_tsne_dict.txt")

    # Running the model
    scores = run_single_model(data=data, **params)
    scores_and_params = {**params, **scores}

    # Saving the results if wished
    if save_in is not None:
        file_exists = Path(save_in).is_file()
        with open(save_in, 'a+') as csvfile:
            writer = DictWriter(csvfile, fieldnames=FIELD_NAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(scores_and_params)
        # read out results if needed
        #results = pd.read_csv(save_in)
        #print(results)

if __name__ == "__main__":
    main()
