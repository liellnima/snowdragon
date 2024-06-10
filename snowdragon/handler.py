from pathlib import Path

from snowdragon.utils.helper_funcs import load_configs
from snowdragon.process.process import preprocess_dataset, preprocess_all_profiles

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
            random_seed: int,
            label_configs: str,
        ):
        """ Initialize snowdragon class object
        """
        self.raw_data_dir = raw_data_dir
        self.exported_smps_dir = exported_smps_dir
        self.smp_npz = smp_npz 
        self.smp_normalized_npz = smp_normalized_npz
        self.preprocess_file = preprocess_file
        self.random_seed = random_seed

        self.label_configs = load_configs(
            config_subdir="graintypes",
            config_name=label_configs,
        )

    
    # 01
    def process(self, process_config: str):
        # load preprocessing configs 
        configs = load_configs(
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
            preprocess_all_profiles(
                data_dir = self.raw_data_dir,
                export_dir = self.exported_smps_dir,
                labels = self.label_configs["labels"],
                npz_name = self.smp_npz,
                export_as = "npz",
                overwrite = False,
                **configs["profile"],
            )

        # second step: processing the whole dataset: 
        # normalize the data, remove nans, sum grains together, etc.
        # if this is done, you can load the data via the output txt file from then on
        preprocess_dataset(
            smp_file_name = self.smp_npz, 
            smp_normalized_file_name = self.smp_normalized_npz,
            output_file = self.preprocess_file,
            random_seed = self.random_seed,
            **configs["dataset"],
        )

    # 02 
    def train(self, train_config: str):
        raise NotImplementedError 

    # 03 
    def validate(self, valid_config: str):
        raise NotImplementedError 

    # 04 
    def test(self, test_config: str):
        raise NotImplementedError

    # 05
    def predict(self, predict_config: str):
        raise NotImplementedError

    def tune(self, tune_config: str):
        raise NotImplementedError