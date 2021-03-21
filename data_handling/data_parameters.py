# Here are all the important parameters for the data preprocssing

# Set folder location of smp data (pnt format)
SMP_LOC = "/home/julia/Documents/University/BA/Data/Arctic/"
# Set file location of temperature data
T_LOC = "/home/julia/Documents/University/BA/Data/Arctic/MOSAiC_ICE_Temperature.csv"
# Set folder name were export files get saved
EXP_LOC = "smp_profiles_test"
# labels for the different grain type markers
LABELS = {"not_labelled": 0, "surface": 1, "ground": 2, "dh": 3, "dhid": 4, "mfdh": 5, "rgwp": 6,
          "df": 7, "if": 8, "ifwp": 9, "sh": 10, "snow-ice": 11, "dhwp": 12, "mfcl": 13, "mfsl": 14, "mfcr": 15, "pp": 16, "rare":17}

# ATTENTION: rare is also added to the dict during preprocessing (sum_up_labels)!
ANTI_LABELS = {0: "not_labelled",  1: "surface", 2: "ground", 3: "dh", 4: "dhid", 5: "mfdh", 6: "rgwp",
          7: "df", 8: "if", 9: "ifwp", 10:"sh", 11: "snow-ice", 12: "dhwp", 13: "mfcl", 14: "mfsl", 15: "mfcr", 16: "pp", 17: "rare"}

COLORS = {0: "dimgray", 1: "chocolate", 2: "darkslategrey", 3: "lightseagreen", 4: "lightsteelblue" , 5: "midnightblue", # "mediumaquamarine"
          6: "tomato", 7: "mediumvioletred", 8: "firebrick", 9: "lightgreen", 10: "orange", 11: "paleturquoise",
          12: "gold", 13: "orchid", 14: "fuchsia", 15: "brown", 16: "green", 17: "blue"}

# arguments for Preprocessing
PARAMS = {"sum_mm": 1, "gradient": True, "window_size": [4,12], "window_type": "gaussian",
          "window_type_std": 1, "rolling_cols": ["mean_force", "var_force", "min_force", "max_force"],
          "poisson_cols": ["median_force", "lambda", "delta", "L"]}
