# retrieve paths that are used throughout the project 
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
SCRIPTS_DIR = ROOT_DIR / "scripts"
STORED_MODELS_DIR = ROOT_DIR / "stored_models"

SRC_DIR = ROOT_DIR / "snowdragon"



