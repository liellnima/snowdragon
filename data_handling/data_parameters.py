# Here are all the important parameters for the data preprocssing

# Set folder location of smp data (pnt format)
SMP_LOC = "/home/julia/Documents/University/BA/Data/Arctic/"
# Set file location of temperature data
T_LOC = "/home/julia/Documents/University/BA/Data/Arctic/MOSAiC_ICE_Temperature.csv"
# Set folder name were export files get saved
EXP_LOC = "smp_profiles"
# labels for the different grain type markers
LABELS = {"not_labelled": 0, "surface": 1, "ground": 2, "dh": 3, "dhid": 4, "mfdh": 5, "rgwp": 6,
          "df": 7, "if": 8, "ifwp": 9, "sh": 10, "snow-ice": 11, "dhwp": 12, "mfcl": 13, "mfsl": 14, "mfcr": 15, "pp": 16}
# arguments for Preprocessing
PARAMS = {"sum_mm": 1, "gradient": True, "window_size": [4,12], "window_type": "gaussian",
          "window_type_std": 1, "rolling_cols": ["mean_force", "var_force", "min_force", "max_force"],
          "poisson_cols": ["median_force", "lambda", "delta"]}
