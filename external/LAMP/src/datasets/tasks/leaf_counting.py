import json
import os

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from config import paths
from src.datasets.dynamic_data import DynamicData

REPOSITORY = "FabrizioDeCastelli/ColoredLeafCounting"
COLORS = {1: "BinaryColor", 3: "ThreeColors"}


class ColoredLeafCountingDataset:
    TASK = "colored_leaf_counting"
    LEVEL = "node"
    SEQUENCE_LEVEL = True

    def __init__(self, name, T, num_colors, **kwargs):
        self.name = name
        self.settings = f"T{T}_K{num_colors}"
        self.sequence_length = T
        self.input_dim = num_colors
        self.output_dim = 1

        setting = f"T={T}-{COLORS[num_colors]}"
        directory = os.path.join(paths.DATA_ROOT, "ColoredLeafCounting")
        spider_file = os.path.join(directory, "spider.pt")
        data_file = os.path.join(directory, setting, "data.pt")
        splits_file = os.path.join(directory, setting, "splits.json")
        if not all(os.path.isfile(f) for f in (spider_file, data_file, splits_file)):
            snapshot_download(
                REPOSITORY,
                repo_type="dataset",
                allow_patterns=[f"{setting}/*", "spider.pt"],
                local_dir=directory,
            )
        spider = torch.load(spider_file, weights_only=False)
        data = torch.load(data_file, weights_only=False)
        with open(splits_file) as f:
            splits = json.load(f)

        train_y = data["y"][splits["train"]].reshape(-1, 1)
        y = (data["y"] - train_y.mean(dim=(0, 1))) / train_y.std(dim=(0, 1))
        root = F.one_hot(data["root_color"], num_classes=num_colors).float()
        samples = []
        for i in range(len(y)):
            x = torch.zeros(data["x"].shape[1], spider["num_nodes"], num_colors)
            if num_colors > 1:
                x[:, spider["root_id"]] = root[i]
            x[:, spider["leaf_ids"]] = data["x"][i]
            samples.append(
                DynamicData(
                    static_keys=("y", "edge_index", "mask"),
                    x=x,
                    y=y[i].reshape(1),
                    edge_index=spider["edge_index"],
                    mask=spider["mask"],
                )
            )
        self.train_data, self.val_data, self.test_data = [
            [samples[i] for i in splits[split]] for split in ("train", "val", "test")
        ]
