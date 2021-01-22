# Here are all the important parameters for the data preprocssing
from pathlib import Path
# Set folder location of smp data (pnt format)
SMP_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/")
# Set file location of temperature data
T_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/MOSAiC_ICE_Temperature.csv")
# Set folder name were export files get saved
EXP_LOC = Path("smp_csv_test04")
# labels for the different grain type markers
LABELS = {"not_labelled": 0, "surface": 1, "ground": 2, "dh": 3, "dhid": 4, "mfdh": 5, "rgwp": 6,
          "df": 7, "if": 8, "ifwp": 9, "sh": 10, "drift_end": 11, "snow-ice": 12}
# arguments for Preprocessing
PARAMS = {"sum_mm": 1, "gradient": True, "window_size": [4,12], "window_type": "gaussian",
          "window_type_std": 1, "rolling_cols": ["mean_force", "var_force", "min_force", "max_force"],
          "poisson_cols": ["median_force", "lambda", "delta"]}
