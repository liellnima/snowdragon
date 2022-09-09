# snowdragon

This repository can be used to run and compare different models for the classification and segmentation of Snow Micro Pen (SMP) profiles.  

The SMP is a fast, high-resolution, portable snow hardness measurement device. The automatic classification and segmentation models can be used for the fast analysis of vast numbers of SMP profiles. For more information about the background of snow layer segmentation and grain type classification please refer to the related publicatoins. Throughout the project the SMP dataset collected during the MOSAiC expedition was used. The plots and results of the different models can be reproduced with this repository.

* Access to raw data: [Available on PANGAEA]()
* Download models: [Available on Zenodo]
* About the SMP: [SLF Website](https://www.slf.ch/en/ueber-das-slf/versuchsanlagen-und-labors/kaeltelabor/snowmicropenr.html)
* About MOSAiC: [MOSAiC Website](https://mosaic-expedition.org/)
* Contact: [julia.kaltenborn@mila.quebec](mailto:julia.kaltenborn@mila.quebec)

## Related publications
* Bsc thesis: "Automatic Snow Classification − A Comparison of Machine Learning Algorithms for the Segmentation and Classification of Snow Micro Penetrometer Profiles" by Julia Kaltenborn
* [“A Comparison of Machine Learning Algorithms for the Segmentation and Classification of Snow Micro Penetrometer Profiles on Arctic Sea Ice”](https://meetingorganizer.copernicus.org/EGU21/EGU21-15637.html), J. Kaltenborn, V. Clay, A. R. Macfarlane, J. M. King, M. Schneebeli, Data Science and Machine Learning for Cryosphere and Climate, EGU General Assembly, 2021. (Abstract)
* [“ML for Snow Stratigraphy Classification”](https://www.climatechange.ai/papers/neurips2021/48), J. Kaltenborn, V. Clay, A. R. Macfarlane and M. Schneebeli, Tackling Climate Change with AI workshop, NeurIPS, 2021. (Presentation & Paper)


## Overview

* ``data/``: Preprocessed SMP profiles as npz files
* ``data_handling/``: Scripts to preprocess and clean the raw SMP profiles
* ``models/``: Scripts to create, use and store models
* ``output/``: Output and results are stored here. Subdir ``evaluation/`` contains plots for each model and profile. ``plots_data/`` contains plots giving an overview over the data. ``plots_results/`` contains plotted results. ``predictions/`` is where predictions are stored. ``scores/`` contains all the scores.

Plots and results with subfolder ``evaluation/`` containing evaluation results
* ``tuning/``: Scripts to tune models
* ``visualization/``: Scripts to create plots

## Setup

This repository runs on Python 3.6. For a quick setup run ``pip install -e .`` The required packages can also be installed with ``pip install -r requirements.txt``. If wished, create an environment beforehand (eg: ``conda create --name=snowdragon python=3.6``).  

The repository does not contain the MOSAiC data used in the related thesis. The data is currently not publicly available, but will become available on 1st January 2023 on the open MCS or [PANGAEA](https://www.pangaea.de/) archives. However, any other SMP dataset can be used as well with this repository.

## Usage

### Prediction with Pretrained Models

If you want to predict some SMP profiles with the models that were trained on the MOSAiC data, the following steps apply:

1. Download the desired models from this [Google Drive Folder](https://drive.google.com/drive/folders/1Rfze6Q95O_zkBbwU67I8eQCl5KFnGayv?usp=sharing).
(Later also from zenodo!)

2. Put the downloaded models into the directory ``snowdragon/models/stored_models/``.

3. Predict all the smp profiles in one directory:

```
python predict.py [arg1] [arg2] #TODO
```

The predicted .ini files can be found in ``snowdragon/output/predictions/MODEL/``.

### Data Preprocessing

To preprocess all SMP profiles, run:

```
python -m data_handling.data_loader [path_npz_file] --smp_src [path_raw_smp_data] --exp_loc [path_dir_smp_preprocessed]
```

* ``[path_npz_file]``: Path and name of the npz file where the complete preprocessed SMP dataset is stored. For example: ``data/all_smp_profiles.npz``
* ``[path_raw_smp_data]``: Path to the directory where the raw SMP data is stored
* ``[path_dir_smp_preprocessed]``: Path to the directory where each single preprocessed SMP profile will be stored. For example: ``data/smp_profiles``

To get information about an already preprocessed data set, run:
```
python -m data_handling.data_loader [path_npz_file] --load_only --test_print
```

For explanations of further preprocessing options run ``python -m data_handling.data_loader -h``.

For smooth default usage, set the ``SMP_LOC`` in ``data_handling/data_parameters.py" to the path where all raw SMP data is stored.

### Tuning

Tuning can be skipped. The default hyperparameters of all models are set to the values which produced the best results for the MOSAiC SMP dataset.

To run tuning, run first model evaluation to create a split up (training, validation, testing) and normalized dataset. The results are saved e.g. in ''data/preprocessed_data_k5.txt``. To tune all models simply run the prepared bash script:

```
bash tuning/tune_models.sh [path_results_csv]
```

``[path_results_csv]`` could be e.g. ``tuning/tuning_results/tuning_run01_all.csv``.

To tune a single model run:

```
python -m tuning.tuning --model_type [wished_model] [path_results_csv]
```
See help options for more information.

After tuning, run ``python -m tuning.eval_tuning`` to aggregate and sort the tuning results for each model. The results are stored in the folder ``tuning/tuning_results/tables``.


### Model Evaluation

Model evaluation consists of preprocessing the complete dataset (in contrast to the single smp profiles as in the first step); evaluating each model; and if desired validating each model.

Preprocessing the complete data set (including data splits and preparing it for model usage) only needs to be done one time:

```
python -m models.run_models --preprocess
```

Afterwards one can just include a flag for evaluating or validating: (All results are stored for each model in the folder ``output/evaluation``.)

```
python -m models.run_models --evaluate --validate
```

Here is the full command, where the smp file and the preprocessed dataset file can be set:

```
python -m --smp_npz [path_npz_file] --preprocess_file [path_txt_file] --preprocess --validate --evaluate
```

* ``[path_npz_file]``: Path to the npz file where the complete SMP dataset was stored. For example: ``data/all_smp_profiles_updated.npz``
* ``[path_txt_file]``: Path to the txt file where the SMP dataset is or will be stored and the different splits of the dataset can be accessed. For example: ``data/preprocessed_data_k5.txt``
* ``preprocess``: Preprocesses the ``[path_npz_file]`` data and stores the split, model-ready data in ``[path_txt_file]``.
* ``evaluate``: Evaluates each model based on the dataset ``[path_txt_file]``. (Go into ``run_models``, function ``evaluate_all_models`` to choose between different models and which evaluation information you want to have from them). Results are stored in ``output/evaluation/``
* ``validate``: Validates each model based on the dataset ``[path_txt_file]``. Results are stored in ``ouput/tables/``.

### Visualization

The data, preprocessing and results are also visualized. The plots are stored in ``outcome`` and can already be found there. There are three sets of plots that can be created: Visualizations of the original data, the normalized data and the results. Look into the code to see which plots are shown and comment out specific plots in ``run_visualization.py`` if desired.

```
python -m visualization.run_visualization --original_data --normalized_data --results
```

## Structure

```
.
├── data
│   └── smp_profiles_updated
├── data_handling
├── models
├── output
│   ├── evaluation
│   │   ├── baseline
│   │   ├── blstm
│   │   ├── bmm
│   │   ├── easy_ensemble
│   │   ├── enc_dec
│   │   ├── gmm
│   │   ├── kmeans
│   │   ├── knn
│   │   ├── label_spreading
│   │   ├── lstm
│   │   ├── rf
│   │   ├── rf_bal
│   │   ├── self_trainer
│   │   └── svm
│   ├── plots_data
│   │   ├── normalized
│   │   └── original
│   ├── plots_results
│   ├── scores
│   └── tables
├── tuning
│   └── tuning_results
│       └── tables
└── visualization
```
