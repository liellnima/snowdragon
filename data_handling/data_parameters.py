# Here are all the important parameters for the data preprocssing

# Set folder location of smp data (pnt format)
SMP_LOC = "/home/julia/Documents/University/BA/Data/Arctic_updated/"
# Set file location of temperature data
T_LOC = "/home/julia/Documents/University/BA/Data/Arctic/MOSAiC_ICE_Temperature.csv"
# Set folder name were export files get saved
EXP_LOC = "data/smp_profiles"
# labels for the different grain type markers
LABELS = {
    "not_labelled": 0, # DO NOT CHANGE THAT ONE
    "surface": 1, # DO NOT CHANGE THAT ONE
    "ground": 2, # DO NOT CHANGE THAT ONE
    "dh": 3,
    "dhid": 4,
    "mfdh": 5,
    "rgwp": 6,
    "df": 7,
    "if": 8,
    "ifwp": 9,
    "sh": 10,
    "snow-ice": 11,
    "dhwp": 12,
    "mfcl": 13,
    "mfsl": 14,
    "mfcr": 15,
    "pp": 16,
    "rare": 17,
}

# ATTENTION: rare is also added to the dict during preprocessing (sum_up_labels)!
ANTI_LABELS = {
    0: "not_labelled", # DO NOT CHANGE THAT ONE
    1: "surface", # DO NOT CHANGE THAT ONE
    2: "ground", # DO NOT CHANGE THAT ONE
    3: "dh",
    4: "dhid",
    5: "mfdh",
    6: "rgwp",
    7: "df",
    8: "if",
    9: "ifwp",
    10: "sh",
    11: "snow-ice",
    12: "dhwp",
    13: "mfcl",
    14: "mfsl",
    15: "mfcr",
    16: "pp",
    17: "rare",
}

ANTI_LABELS_LONG = {
    0: "Not labelled", # DO NOT CHANGE THAT ONE
    1: "Surface", # DO NOT CHANGE THAT ONE
    2: "Ground", # DO NOT CHANGE THAT ONE
    3: "Depth Hoar",
    4: "Depth Hoar\nIndurated",
    5: "Melted Form\nDepth Hoar",
    6: "Rounded Grains\nWind Packed",
    7: "Decomposed\nand Fragmented\nPrecipitation Particles",
    8: "Ice Formation",
    9: "Ice Formation\nWind Packed",
    10: "Surface Hoar",
    11: "Snow Ice",
    12: "Depth Hoar\nWind Packed",
    13: "Melted Form\nClustered Rounded Grains",
    14: "Melted Form\nSlush",
    15: "Melt-freeze\nCrust",
    16: "Precipitation\nParticles",
    17: "Rare",
}

COLORS = {
    0: "dimgray",
    1: "chocolate",
    2: "darkslategrey",
    3: "lightseagreen", #20b2aa
    4: "lightsteelblue", #b0c4de
    5: "midnightblue", #191970 # "mediumaquamarine"
    6: "tomato", #ff6347
    7: "mediumvioletred",
    8: "firebrick",
    9: "lightgreen",
    10: "orange",
    11: "paleturquoise", #afeeee
    12: "gold", #ffd700
    13: "orchid",
    14: "fuchsia",
    15: "saddlebrown",
    16: "green", #2ca02c
    17: "blue", #0000ff
}

# arguments for Preprocessing
PARAMS = {
    "sum_mm": 1,
    "gradient": True,
    "window_size": [4, 12],
    "window_type": "gaussian",
    "window_type_std": 1,
    "rolling_cols": ["mean_force", "var_force", "min_force", "max_force"],
    "poisson_cols": ["median_force", "lambda", "delta", "L"],
}

# Colors for the different models
# TODO: Pick good colors
MODEL_COLORS02 = {
    "Majority Vote": "grey",
    "K-means": "fuchsia",
    "Gaussian Mixture Model": "orchid",
    "Bayesian Gaussian Mixture Model": "palevioletred",
    "Self Trainer": "plum",
    "Label Propagation": "blueviolet",
    "Random Forest": "indigo",
    "Balanced Random Forest": "darkblue",
    "Support Vector Machine": "blue",
    "K-nearest Neighbors": "royalblue",
    "Easy Ensemble": "dodgerblue",
    "LSTM": "green",
    "BLSTM": "springgreen",
    "Encoder Decoder": "lime",
}

MODEL_LABELS = {
    "baseline": "Majority Vote",
    "bmm": "Bayesian Gaussian Mixture Model",
    "gmm": "Gaussian Mixture Model",
    "kmeans": "K-means",
    "self_trainer": "Self Trainer",
    "label_spreading": "Label Propagation",
    "easy_ensemble": "Easy Ensemble",
    "knn": "K-nearest Neighbors",
    "rf": "Random Forest",
    "rf_bal": "Balanced Random Forest",
    "svm": "Support Vector Machine",
    "lstm": "LSTM",
    "blstm": "BLSTM",
    "enc_dec": "Encoder Decoder"
}

# experimenting with the colors
# group 01: GRAY - baseline
# group 02: ORANGE/RED semi-supervised
#   kmeans, gmm, bgm, self-trainer, label propagation
# group 03: BLUE supervised
#   random forest, rf balanced, svm, knn, easy ensemble
# group 04: GREEN ANNs
#   lstm, blstm, enc dec
MODEL_COLORS = {
    "Majority Vote": "grey",
    "K-means": "xkcd:dandelion",
    "Gaussian Mixture Model": "xkcd:golden rod",
    "Bayesian Gaussian Mixture Model": "xkcd:orange",
    "Self Trainer": "xkcd:red",
    "Label Propagation": "xkcd:crimson",
    "Random Forest": "indigo",
    "Balanced Random Forest": "navy",
    "Support Vector Machine": "blue",
    "K-nearest Neighbors": "xkcd:sky blue",
    "Easy Ensemble": "xkcd:light teal",
    "LSTM": "xkcd:green apple", #"xkcd:slime green" #apple
    "BLSTM": "xkcd:emerald",
    "Encoder Decoder": "xkcd:forest green",
}

# Selection of snow layer types that we use
# TODO: Turn into dictionary with colors?
# Sorted by amount of examples?
SNOW_TYPES_SELECTION = ["rgwp", "dh", "dhid", "dhwp", "mfdh", "pp", "rare"]
