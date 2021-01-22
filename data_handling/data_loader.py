from data_preprocessing import export_pnt, npz_to_pd, idx_to_int
from data_parameters import SMP_LOC, EXP_LOC, PARAMS
from pathlib import Path
import numpy as np
import time

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

def preprocess_data(data_dir, export_dir, united_npz_name="smp_all.npz", overwrite=False,**PARAMS):
    """ Preprocesses the pnt data in the data_dir according the given parameters.
    Data is intermittently exported as single npz files. The single npz_files are combined to one pandas frame and exported as one united npz file.
    Parameters:
        data_dir (Path): Directory where all the pnt data (= the smp profiles) and their ini files (= the smp markers) are stored
        export_dir (Path): Directory where the exported npz files (one per each smp profile) is saved
        united_npz_name (String): how the npz file should be called where all the smp profiles are stored together (must end with .npz)
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
    np.savez_compressed(united_npz_name, **dict)

    end = time.time()
    print("Elapsed time for export and dataframe creation: ", end-start)
    print("\nUnited smp data was stored in {}.".format(united_npz_name))

def load_data(npz_file, test=False):
    """ Wrapper Function for npz_to_pd for easier usage. Returns the data from a npz file as pd.DataFrame.
    Paramters:
        npz_file (String): Name of the npz file to load
        test (Boolean): Default false. Indicates if some information should be printed out about the data. Can be used for testing purposes.
    Returns:
        pd.DataFrame: the data from the npz file loaded into a dataframe
    """
    if test:
        smp = npz_to_pd(npz_file, is_dir=False)
        print_test_df(smp)
        return smp
    else:
        return npz_to_pd(npz_file, is_dir=False)

#TODO: make this user friendly - use this with commandline

def main():
    preprocess_data(data_dir=SMP_LOC, export_dir=EXP_LOC, united_npz_name="smp_lambda_delta_gradient.npz", overwrite=False, **PARAMS)
    smp = load_data(npz_file="smp_lambda_delta_gradient.npz", test=True)


if __name__ == "__main__":
    main()
