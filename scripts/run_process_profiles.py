from snowdragon.utils.helper_funcs import load_smp_data
from snowdragon.process.process import preprocess_all_profiles
#from snowdragon.process.archived_data_parameters import SMP_LOC, EXP_LOC, PARAMS
from pathlib import Path
import argparse

# Example of how to use the parser to export everything
# python scripts/run_process_profiles.py data/all_smp_profiles.npz --smp_src /home/julia/Documents/University/BA/Data/Arctic/ --exp_loc data/smp_profiles

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
parser.add_argument("--gradient", default=int(PARAMS["gradient"]), type=int, help="Should the gradient be excluded from the formed dataset? Use 0 for False and 1 for True.")
parser.add_argument("--window_size", default=PARAMS["window_size"], type=list, help="List of window sizes that should be applied for rolling window. e.g. [4]")
parser.add_argument("--window_type", default=PARAMS["window_type"], type=str, help="Window type of the rolling window")
parser.add_argument("--window_type_std", default=PARAMS["window_type_std"], type=int, help="std used for window type")
parser.add_argument("--rolling_cols", default=PARAMS["rolling_cols"], type=list, help="List of columns over which should be rolled")
parser.add_argument("--poisson_cols", default=PARAMS["poisson_cols"], type=list, help="List of features that should be taken from poisson shot model. List can include: distance, median_force, lambda, f0, delta, L")


def main():
    args = parser.parse_args()
    params = vars(args)
    print("Parameters used are:\n", params)
    # parse smp_src and exp_loc to Paths
    data_dir = Path(params["smp_src"])
    exp_loc = Path(params["exp_loc"])

    if not params["load_only"]:
        preprocess_all_profiles(data_dir=data_dir, export_dir=exp_loc, **params)
    smp = load_smp_data(**params)
    if params["load_only"]: smp.info()


if __name__ == "__main__":
    main()
