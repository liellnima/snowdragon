import pickle
import tabulate
import pandas as pd
from pathlib import Path

from utils.helper_funcs import load_configs, load_results
from process.process import preprocess_dataset, preprocess_all_profiles
from ml.run_models import validate_all_models, train_and_store_models, evaluate_all_models

class Snowdragon():
    """
    """
    # 00
    def __init__(
            self,  
            raw_data_dir: Path, 
            exported_smps_dir: Path,
            smp_npz: Path, 
            smp_normalized_npz: Path, 
            preprocess_file: Path,
            models: list,
            random_seed: int,
            config_files: dict,
        ):
        """ Initialize snowdragon class object
        """
        self.raw_data_dir = raw_data_dir
        self.exported_smps_dir = exported_smps_dir
        self.smp_npz = smp_npz 
        self.smp_normalized_npz = smp_normalized_npz
        self.preprocess_file = preprocess_file
        self.models = list
        self.random_seed = random_seed

        self.label_configs = load_configs(
            config_subdir="graintypes",
            config_name=config_files["graintypes"],
        )

        self.color_configs = load_configs(
            config_subdir="colors",
            config_name=config_files["colors"],
        )

        self.visualize_configs = load_configs(
            config_subdir="visualize",
            config_name=config_files["visualize"],
        )

        self.data = None
    
    # 01 A process data
    # TODO make this pretty, put it into a class
    def process(self, process_config: str):
        # load preprocessing configs 
        preprocessing_configs = load_configs(
            config_subdir="preprocessing",
            config_name=process_config,
        )

        # TODO check if the smp npz file already exists
        npz_exists = self.smp_npz.is_file()

        # first step: processing the raw smp profiles: 
        # summing to 1mm, applying moving windows, 
        # and handle everything on profile level.
        # The results are stored in an npz file. If that file
        # already exists, this step is skipped.
        if not npz_exists:
            print("Preprocess all Profiles: \n")
            preprocess_all_profiles(
                data_dir = self.raw_data_dir,
                export_dir = self.exported_smps_dir,
                labels = self.label_configs["labels"],
                npz_name = self.smp_npz,
                export_as = "npz",
                overwrite = False,
                **preprocessing_configs["profile"],
            )

        # second step: processing the whole dataset: 
        # normalize the data, remove nans, sum grains together, etc.
        # if this is done, you can load the data via the output txt file from then on
        print("Preprocess the Dataset: \n")
        self.data = preprocess_dataset(
            smp_file_name = self.smp_npz, 
            smp_normalized_file_name = self.smp_normalized_npz,
            output_file = self.preprocess_file,
            random_seed = self.random_seed,
            label_configs = self.label_configs,
            color_configs = self.color_configs,
            visualize_configs = self.visualize_configs,
            **preprocessing_configs["dataset"],
        )
        print("Done.")

    # 01 B: load data that has been preprocessed
    def load_processed_data(self):
        with open(self.preprocess_file, "rb") as f:
            data = pickle.load(f)
        
    # 02 
    def train(self, train_config: str):
        raise NotImplementedError 

    # 03 
    # TODO make this pretty, put it into a class
    def validate(self, valid_config: str):
        intermediate_results = "data/validation_results.txt"
        validate_all_models(self.data, intermediate_results)

        all_scores = load_results(intermediate_results)

        # print and save results the validation results
        all_scores = pd.DataFrame(all_scores).rename(columns={"test_balanced_accuracy": "test_bal_acc",
                                                             "train_balanced_accuracy": "train_bal_acc",
                                                             "test_recall": "test_rec",
                                                             "train_recall": "train_rec",
                                                             "test_precision": "test_prec",
                                                             "train_precision": "train_prec",
                                                             "train_roc_auc": "train_roc",
                                                             "test_roc_auc": "test_roc",
                                                             "train_log_loss": "train_ll",
                                                             "test_log_loss": "test_ll"})
        print(tabulate(pd.DataFrame(all_scores), headers="keys", tablefmt="psql"))

        with open("output/tables/models_160smp_test01.txt", 'w') as f:
            f.write(tabulate(pd.DataFrame(all_scores), headers="keys", tablefmt="psql"))

        with open("output/tables/models_160smp_test01_latex.txt", 'w') as f:
            f.write(tabulate(pd.DataFrame(all_scores), headers="keys", tablefmt="latex_raw"))


    # 04 
    def test(self, test_config: str):
        raise NotImplementedError

    # 05
    def predict(self, predict_config: str):
        raise NotImplementedError

    # 06
    def tune(self, tune_config: str):
        raise NotImplementedError
    
    # TODO delete, separate things out
    def train_and_store(self):
        train_and_store_models(self.data, models=self.models)

    # TODO delete, separate things out
    def evaluate(self):
        evaluate_all_models(self.data, models=self.models, overwrite_tables=False)