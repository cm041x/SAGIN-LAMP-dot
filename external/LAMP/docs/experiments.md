# Running experiments

An experiment has two stages: model selection trains every configuration of a
hyperparameter grid and picks the best on the validation set; model assessment
retrains the selected configuration with several seeds and reports the mean and
std on the test set.

## Model selection

```bash
python main.py --task leaf_counting   --dataset SpiderW5H10
python main.py --task node_prediction --dataset ChickenPox
python main.py --task link_prediction --dataset As-733
python main.py --task leaf_counting   --dataset SpiderW5H10 --model DCRNN
```

`main.py` trains every configuration of the grid in
`configurations/hyperparameters/<task>/<dataset>/<model>.yaml` on every setting
of `configurations/datasets/<task>/<dataset>.yaml`, and writes the index of the
configuration with the best validation score to `best_run.txt`. `--model`
defaults to LAMP.

## Model assessment

```bash
python run_reinit.py --task node_prediction --dataset ChickenPox
python run_reinit.py --task node_prediction --dataset ChickenPox --config my_config.yaml
```

`run_reinit.py` retrains the configuration selected by `main.py` with
`N_REINIT` seeds (`seed`, `seed + 1`, ...) and writes the mean and std of every
score to `scores.csv`. `--config` assesses a configuration of your choice
instead: a YAML file shaped like a grid, with a single value per key.

## Configuration files

**Datasets**, `configurations/datasets/<task>/<dataset>.yaml`. Top-level keys
choose the dataset; a list defines one setting per value, e.g.
`T: [200, 400, 600]` and `num_colors: [1, 3]` give the six ColoredLeafCounting
settings. The `default` block holds the training defaults:

| Key | Meaning |
|---|---|
| `seed` | seed of model selection; assessment uses `seed`, `seed + 1`, ... |
| `N_REINIT` | number of assessment runs |
| `criterion` | training loss: `MSE`, `MAE` or `BinaryClassificationLoss` |
| `metrics` | scores written to `scores.csv`: `MSE`, `MAE`, `F1`, `RocAuc` |
| `epochs`, `patience`, `min_delta` | training length and early stopping |
| `batch_size` | sequences (ColoredLeafCounting) or snapshots per batch |

**Grids**, `configurations/hyperparameters/<task>/<dataset>/<model>.yaml`:

```yaml
Model:
  name: LAMP
  hidden_dim: 32
  beta: [-1.0, 0.1, 1.0]
  gamma: [0.01, 0.1]
  ...
Optimizer:
  name: Adam
  learning_rate: [0.01, 0.001]
  weight_decay: 0.0001
Override:
  batch_size: [32, 64, 128]
Validation:
  monitor: MSE
```

Every list is an axis of the grid, and the grid is their cartesian product.
`Override` sets `batch_size`, `epochs` or `patience` instead of the dataset's
`default`. Each run keeps the epoch with the lowest validation loss; `monitor`
is the validation score that then picks the best configuration of the grid
(`F1` and `RocAuc` are maximized, the others minimized).

The models are `LAMP` and the baselines `TGCN`, `DCRNN`, `GCRN_LSTM`,
`GCRN_GRU`, `A3TGCN`, `DynGESN`, `GCLSTM`, `EvolveGCN_O` and `EvolveGCN_H`. The
baselines have grids for ColoredLeafCounting only.

## Devices

Runs are spread over every visible GPU with Ray. These options apply to
`main.py`, `run_reinit.py` and `paper.reproduce`:

| Option | Effect |
|---|---|
| `--gpus 0 2` | use these GPUs (default: all visible) |
| `--workers 3` | train three runs at a time on each GPU |
| `--cpu` | train on CPU, one run at a time (the default without a GPU) |
| `--deterministic` | deterministic kernels: a rerun on the same hardware gives exactly the same numbers, about 1.7× slower on GPU |

Runs that already finished with the same configuration, seed and
`--deterministic` flag are skipped, so an interrupted command can be relaunched.

Results go to `experiments/tuning/...` and `experiments/reinit/...`; see
[results.md](results.md).
