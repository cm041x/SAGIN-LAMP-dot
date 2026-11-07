import os

PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.environ.get("LAMP_DATA_ROOT", os.path.join(PROJECT_FOLDER_PATH, "data"))

EXPERIMENTS_PATH = os.path.join(PROJECT_FOLDER_PATH, "experiments")

CONFIGURATIONS_PATH = os.path.join(PROJECT_FOLDER_PATH, "configurations")
DATASET_SPECIFIC_HYPERPARAMETERS_PATH = os.path.join(CONFIGURATIONS_PATH, "datasets")
TUNING_HYPERPARAMETERS_PATH = os.path.join(CONFIGURATIONS_PATH, "hyperparameters")
