import os

import torch
from huggingface_hub import snapshot_download

from config import paths

REPOSITORY = "FabrizioDeCastelli/temporal-link-level-prediction"


class LinkPredictionDataset:
    TASK = "link_prediction"
    LEVEL = "link"
    SEQUENCE_LEVEL = False

    def __init__(self, name, **kwargs):
        self.name = name
        self.settings = "default"
        directory = os.path.join(paths.DATA_ROOT, "temporal-link-level-prediction")
        files = [os.path.join(directory, name, f"{split}.pt") for split in ("train", "val", "test")]
        if not all(os.path.isfile(file) for file in files):
            snapshot_download(
                REPOSITORY, repo_type="dataset", allow_patterns=[f"{name}/*"], local_dir=directory
            )
        self.train_data, self.val_data, self.test_data = [
            torch.load(file, weights_only=False) for file in files
        ]
        self.input_dim = self.train_data[0].x.shape[-1]
        self.output_dim = 1
