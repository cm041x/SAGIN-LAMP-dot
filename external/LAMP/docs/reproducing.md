# Reproducing the paper

Every table of the paper has a file in [`paper/`](../paper) that lists, for each
entry, the selected hyperparameters, the seeds of the reported runs and the
value printed in the paper. One command retrains all its runs and prints the
mean ± std of the reruns next to the paper's:

```bash
python -m paper.reproduce paper/clc.yaml
```

```
                  mean     std  paper mean  paper std
entry   metric
T200_K1 MSE       ...     ...       0.467      0.083
...
```

| Paper | Command |
|---|---|
| Tables 1–2: ColoredLeafCounting | `python -m paper.reproduce paper/clc.yaml` |
| Table 3: dissipation and weight-constraint ablation | `python -m paper.reproduce paper/clc_ablation.yaml` |
| Table 4: Wikipedia Math, Twitter Tennis, ChickenPox | `python -m paper.reproduce paper/node_prediction.yaml` |
| Tables 6 and 8: AS-733, Bitcoin-α | `python -m paper.reproduce paper/link_prediction.yaml` |
| Table 7: runtimes | `python -m paper.runtimes` |
| Table 9: sensitivity to ε, L, β on Twitter Tennis | `python -m paper.reproduce paper/twitter_tennis_sensitivity.yaml` |
| Figure 2: Jacobian eigenvalues on ChickenPox | `python -m paper.figures eigenvalues paper/node_prediction.yaml ChickenPox` |
| Figure 3a: stability ratio on ChickenPox | `python -m paper.figures stability paper/node_prediction.yaml ChickenPox` |
| Figure 3b: stability ratio on three-colour T=400 | `python -m paper.figures stability paper/clc.yaml T400_K3` |
| Figure 4a: input-to-state stability on ChickenPox | `python -m paper.figures iss paper/node_prediction.yaml ChickenPox` |
| Figure 4b: input-to-state stability on three-colour T=400 | `python -m paper.figures iss paper/clc.yaml T400_K3` |

## Options

- `--entries T200_K1 T600_K3` retrains only some entries of a table.
- `--gpus`, `--workers`, `--cpu` and `--deterministic` choose where and how the
  runs train, see [Devices](experiments.md#devices).
- Runs that already finished are skipped, so an interrupted command can simply
  be relaunched.

The figure scripts load the weights saved by `paper.reproduce`, so run the
corresponding table first. `--run 2` uses the third seed of the entry instead of
the first; `--steps` and `--max-scale` set the input scales of the ISS curve.

`paper.runtimes` times training and inference epochs on ColoredLeafCounting;
`--entries`, `--epochs`, `--batch-size` and `--cpu` change what it measures.

## Outputs

```
experiments/paper/<table>/
  results.csv                 mean and std of every entry next to the paper's
  <entry>/run_<i>/
    scores.csv                train, validation and test scores of the best epoch
    hyperparameters.json      the exact configuration and seed
    model.pt                  the trained weights
    figures/                  written by paper.figures
```

## Table files

A table file names the task and, unless every entry sets its own, the
dataset, and the `monitor` that selected the configurations. Each entry gives
the dataset setting, the seeds, the paper's mean and std, and the
configuration:

```yaml
task: leaf_counting
dataset: SpiderW5H10
monitor: MSE

entries:
  T200_K1:
    setting: {T: 200, num_colors: 1}
    seeds: [9, 90, 900, 9000, 90000]
    paper: {MSE: [0.467, 0.083]}
    Model: {name: LAMP, hidden_dim: 32, L: 1, xi: 1.0, beta: 1.0, gamma: 0.01, ...}
    Optimizer: {name: Adam, learning_rate: 0.001, weight_decay: 0.0001}
    Override: {batch_size: 64}
```

`Override` sets `batch_size`, `epochs` or `patience` instead of the
dataset's `default`.
