import yaml
import argparse 

from pathlib import Path 

from snowdragon import ROOT_DIR
from snowdragon.handler import Snowdragon

parser = argparse.ArgumentParser(description="Snowdragon Runner.")
parser.add_argument("--configs", "-c", default="configs/main_config.yaml", type=str, help="configurations for snowdragon runner")

if __name__ == "__main__":
    """ Execute main functions.
    """
    # read in configs 
    args = parser.parse_args()
    configs = None
    with open(args.configs) as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            print(err)

    raw_data_dir = Path(configs["raw_data"]["smp"])

    preprocessed_data = ROOT_DIR / configs["processed_data"]["preprocessed_data"]
    exported_smps_dir = ROOT_DIR / Path(configs["processed_data"]["exported_smp_files"])
    normalized_npz_file = ROOT_DIR / configs["processed_data"]["normalized_npz_file"]
    npz_file = ROOT_DIR / configs["processed_data"]["npz_file"]
    evaluation_dir = ROOT_DIR / configs["processed_data"]["evaluation"]

    random_seed = configs["random_seed"]

    snowdragon = Snowdragon(
        raw_data_dir = raw_data_dir,
        exported_smps_dir = exported_smps_dir,
        smp_npz = npz_file,
        smp_normalized_npz = normalized_npz_file,
        preprocess_file = preprocessed_data,
        random_seed = random_seed,
        label_configs = configs["configs"]["graintypes"],
    )

    if configs["run"]["preprocess"]:
        snowdragon.process(
            process_config = configs["configs"]["preprocessing"]
        )

    if configs["run"]["train"]:
        snowdragon.train() 

    if configs["run"]["validate"]:
        snowdragon.validate()

    if configs["run"]["test"]:
        snowdragon.test() 

    if configs["run"]["predict"]:
        snowdragon.predict()

    if configs["run"]["tune"]:
        snowdragon.tune()