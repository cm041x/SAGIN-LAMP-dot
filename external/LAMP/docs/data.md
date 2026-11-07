# Data

Datasets download automatically the first time they are used, into `data/`.
Set `LAMP_DATA_ROOT=/some/path` to keep them elsewhere, for example on a shared
disk.

| Dataset | Source | Stored in |
|---|---|---|
| ColoredLeafCounting | [FabrizioDeCastelli/ColoredLeafCounting](https://huggingface.co/datasets/FabrizioDeCastelli/ColoredLeafCounting) | `data/ColoredLeafCounting/` |
| Wikipedia Math, ChickenPox Hungary, Twitter Tennis | `torch_geometric_temporal` | `data/node_prediction/<name>/` |
| AS-733, Bitcoin-α | [FabrizioDeCastelli/temporal-link-level-prediction](https://huggingface.co/datasets/FabrizioDeCastelli/temporal-link-level-prediction) | `data/temporal-link-level-prediction/<name>/` |

**ColoredLeafCounting** is the long-range benchmark introduced in the paper.
The Hub holds the exact sequences and splits of the paper for six settings,
T ∈ {200, 400, 600} × {binary, three colours}; only the setting you use is
downloaded. The targets are standardized with the mean and standard deviation
of the training split.

**Node prediction.** Each dataset is a single sequence of snapshots, split in
time: the first 80% for training, the next 10% for validation, the last 10% for
testing. The split is cached as `train.pt`, `val.pt` and `test.pt`.

**Link prediction.** The Hub files are already split, with the negative edges
included, so every run sees the same negatives.

To run without network access, copy a `data/` folder that already holds the
datasets you need.
