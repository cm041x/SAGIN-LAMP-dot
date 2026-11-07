import os

import torch
from torch_geometric_temporal.dataset import (
    ChickenpoxDatasetLoader,
    TwitterTennisDatasetLoader,
    WikiMathsDatasetLoader,
)

from config import paths
from src.datasets.dynamic_data import DynamicData

LOADERS = {
    "ChickenPox": ChickenpoxDatasetLoader,
    "TwitterTennis": TwitterTennisDatasetLoader,
    "WikiMat": WikiMathsDatasetLoader,
}


class NodePredictionDataset:
    TASK = "node_prediction"
    LEVEL = "node"
    SEQUENCE_LEVEL = False

    def __init__(self, name, val_split, test_split, **kwargs):
        self.name = name
        self.settings = "default"
        directory = os.path.join(paths.DATA_ROOT, "node_prediction", name)
        files = [os.path.join(directory, f"{split}.pt") for split in ("train", "val", "test")]
        if not all(os.path.isfile(file) for file in files):
            snapshots = list(LOADERS[name]().get_dataset())
            n_val = int(len(snapshots) * val_split)
            n_test = int(len(snapshots) * test_split)
            n_train = len(snapshots) - n_val - n_test
            x = torch.stack([s.x for s in snapshots])
            y = torch.stack([s.y for s in snapshots])
            bounds = [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, len(snapshots))]
            os.makedirs(directory, exist_ok=True)
            for file, (start, end) in zip(files, bounds):
                split = DynamicData(
                    x=x[start:end],
                    y=y[start:end],
                    edge_index=[s.edge_index for s in snapshots[start:end]],
                    edge_attr=[s.edge_attr for s in snapshots[start:end]],
                )
                torch.save(split, file)
        self.train_data, self.val_data, self.test_data = [
            torch.load(file, weights_only=False) for file in files
        ]
        self.input_dim = self.train_data[0].x.shape[-1]
        self.output_dim = 1
