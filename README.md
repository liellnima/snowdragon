# snowdragon

This repository can be used to run and compare different models for the classification and segmentation of Snow Micro Pen (SMP) profiles.  

The SMP is a fast, high-resolution, portable snow hardness measurement device. The automatic classification and segmentation models can be used for the fast analysis of vast numbers of SMP profiles. For more information about the background of snow layer segmentation and grain type classification please refer to the related thesis. In the thesis the SMP dataset collected during the MOSAiC expedition was used. The plots and results of the different models can be reproduced with this repository.

* Related thesis: "Automatic Snow Classification − A Comparison of Machine Learning Algorithms for the Segmentation and Classification of Snow Micro Penetrometer Profiles" by Julia Kaltenborn
* About the SMP: [SLF Website](https://www.slf.ch/en/ueber-das-slf/versuchsanlagen-und-labors/kaeltelabor/snowmicropenr.html) 
* About MOSAiC: [MOSAiC Website](https://mosaic-expedition.org/)
* Contact: [jkaltenborn@uos.de](mailto:jkaltenborn@uos.de)


## Overview

* ``data/``: Preprocessed SMP profiles as npz files
* ``data_handling/``: Scripts to preprocess and clean the raw SMP profiles
* ``models/``: Scripts to create and use models
* ``plots/``: Plots and results with subfolder ``evaluation/`` containing evaluation results
* ``tuning/``: Scripts to tune models

## Setup

This repository runs on Python 3.6. The required packages can be installed with ``pip install -r requirements.txt``. If wished, create an environment beforehand.  

The repository does not contain the MOSAiC data used in the related thesis. The data is currently not publicly available, but will become available on 1st January 2023 on the open MCS or [PANGAEA](https://www.pangaea.de/) archives. However, any other SMP dataset can be used as well with this repository.

## Usage

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

Model evaluation consists of preprocessing the complete dataset, and producing evaluation results for each model.

For preprocessing the first time, go into the main of the file and set ``smp_file_name`` and ``output_file`` to the appropiate value, and set ``data_dict`` to None. For example:

```
smp_file_name = "data/all_smp_profiles.npz"
output_file = "data/preprocessed_data_k5.txt"
data_dict = None
```
After preprocessing the data for the first time, ``data_dict`` can be set to the ``output_file`` value and preprocessing will be skipped from then on.

To run prepocessing and evaluation, run:

```
python -m models.run_models
```
If ``visualize`` is set to ``True`` the original and preprocessed data will be visualized. The plots are not saved, but all plots were already stored and can be found in the plots folder.

After preprocessing all models are evaluated. All results are stored for each model in the folder ``plots/evaluation``.

## Structure

```
.
├── data
│   └── smp_profiles_updated
├── data_handling
├── models
├── plots
│   ├── data_original
│   ├── data_preprocessed
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
│   │   ├── svm
│   │   └── trues
│   ├── label_frequency
│   ├── other
│   │   └── beautiful
│   └── tables
└── tuning
    └── tuning_results
        └── tables

```
