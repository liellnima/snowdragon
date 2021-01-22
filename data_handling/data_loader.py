from data_handling.data_preprocessing import export_pnt, npz_to_pd, idx_to_int
from data_handling.data_parameters import SMP_LOC, EXP_LOC, PARAMS
from pathlib import Path
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description="Exporting SMP Profiles. Unifying SMP pnt files into one npz file. Loading united npz file. All arguments have default parameters from the data_parameters.py file.")

# Mandatory argument
parser.add_argument("npz_name", type=str, help="Name of the united npz file")

# loading and exporting option
parser.add_argument("--overwrite", action="store_true", help="If overwrite is set, the data in the exp_loc will be overwritten. If not set, already existing files will be skipped during exportation and preprocessing.")
parser.add_argument("--test_print", action="store_true", help="If true, some information of the unite smp dataframe will be printed.")
parser.add_argument("--load_only", action="store_true", help="In case of load only, no data is exported, but only loaded from the given npz_name file.")

# Data directory arguments
parser.add_argument("--smp_src", default=SMP_LOC, type=str, help="The directory where the pnt and ini files of the smp profiles are.")
parser.add_argument("--exp_loc", default=EXP_LOC, type=str, help="The directory where the single npz files of the smp profiles should be saved.")

# Parameter arguments for data processing
parser.add_argument("--sum_mm", default=PARAMS["sum_mm"], type=float, help="How many mm of the smp data should be summed up together? (e.g. 1 -> resolution of 1mm snow layers)")
parser.add_argument("--gradient", default= PARAMS["gradient"], type=bool, help="Should the gradient be included in the formed dataset?")
parser.add_argument("--window_size", default=PARAMS["window_size"], type=list, help="List of window sizes that should be applied for rolling window. e.g. [4]")
parser.add_argument("--window_type", default=PARAMS["window_type"], type=str, help="Window type of the rolling window")
parser.add_argument("--window_type_std", default=PARAMS["window_type_std"], type=int, help="std used for window type")
parser.add_argument("--rolling_cols", default=PARAMS["rolling_cols"], type=list, help="List of columns over which should be rolled")
parser.add_argument("--poisson_cols", default=PARAMS["poisson_cols"], type=list, help="List of features that should be taken from poisson shot model. List can include: distance, median_force, lambda, f0, delta, L")

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

def preprocess_data(data_dir, export_dir, npz_name="smp_all.npz", overwrite=False,**PARAMS):
    """ Preprocesses the pnt data in the data_dir according the given parameters.
    Data is intermittently exported as single npz files. The single npz_files are combined to one pandas frame and exported as one united npz file.
    Parameters:
        data_dir (Path): Directory where all the pnt data (= the smp profiles) and their ini files (= the smp markers) are stored
        export_dir (Path): Directory where the exported npz files (one per each smp profile) is saved
        npz_name (String): how the npz file should be called where all the smp profiles are stored together (must end with .npz)
        overwrite (Boolean): Default False. If the smp profiles were already exported once into the export_dir,
            this data can be overwritting by setting overwrite = True. Otherwise only those files will be exported that do not exist yet.
        **PARAMS:
            sum_mm (num): How many mm of the smp data should be summed up together? (e.g. 1 -> resolution of 1mm snow layers)
            gradient (Boolean): Should the gradient be included in the formed dataset?
            window_size (list): Window sizes that should be applied during preprocessing
            window_type (String): Window type of the rolling window
            window_type_std (int): std used for window type
            rolling_cols (list): List of columns over which should be rolled
            poisson_cols (list): List of features that should be taken from poisson shot model.
                List can include: "distance", "median_force", "lambda", "f0", "delta", "L"
    """
    print("Starting to export and/or convert data")
    # export, unite and label smp data
    start = time.time()
    # export data from pnt to csv or npz
    export_pnt(pnt_dir=data_dir, target_dir=export_dir, export_as="npz", overwrite=overwrite, **PARAMS)
    # load pd.DataFrame from all npz files and save this pd as united DataFrame in npz
    smp_first = npz_to_pd(export_dir, is_dir=True)
    dict = smp_first.to_dict(orient="list")
    np.savez_compressed(npz_name, **dict)

    end = time.time()
    print("Elapsed time for export and dataframe creation: ", end-start)
    print("\nUnited smp data was stored in {}.".format(npz_name))

def load_data(npz_name, test_print=False, **kwargs):
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

#TODO: make this user friendly - use this with commandline

def main():
    args = parser.parse_args()
    params = vars(args)
    print("Parameters used are:\n", params)
    # parse smp_src and exp_loc to Paths
    data_dir = Path(params["smp_src"])
    exp_loc = Path(params["exp_loc"])

    if not params["load_only"]:
        preprocess_data(data_dir=data_dir, export_dir=exp_loc, **params)
    smp = load_data(**params)


if __name__ == "__main__":
    main()
