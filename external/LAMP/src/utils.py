import os
from itertools import product

import yaml

from config import paths
from src.datasets.tasks.leaf_counting import ColoredLeafCountingDataset
from src.datasets.tasks.link_prediction import LinkPredictionDataset
from src.datasets.tasks.node_prediction import NodePredictionDataset

DATASETS = {
    "leaf_counting": ColoredLeafCountingDataset,
    "node_prediction": NodePredictionDataset,
    "link_prediction": LinkPredictionDataset,
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cartesian_product(data):
    lists = {k: v for k, v in data.items() if isinstance(v, list) and k != "default"}
    nested = {k: v for k, v in data.items() if isinstance(v, dict) and k != "default"}
    fixed = {k: v for k, v in data.items() if k not in lists and k not in nested}
    combinations = [{**dict(zip(lists, values)), **fixed} for values in product(*lists.values())]
    for key, value in nested.items():
        combinations = [
            {**combination, key: inner}
            for combination in combinations
            for inner in cartesian_product(value)
        ]
    return combinations


def dataset_settings(task, name):
    path = os.path.join(paths.DATASET_SPECIFIC_HYPERPARAMETERS_PATH, task, f"{name}.yaml")
    return cartesian_product(load_config(path))


def prepare_dataset(task, dataset_config):
    return DATASETS[task](**dataset_config)


def run_dir(stage, model_name, dataset):
    return os.path.join(
        paths.EXPERIMENTS_PATH, stage, dataset.TASK, dataset.name, dataset.settings, model_name
    )
