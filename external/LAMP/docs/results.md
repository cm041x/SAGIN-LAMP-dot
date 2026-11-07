# Collecting results

Every run writes its own folder under `experiments/`:

```
experiments/
  tuning/<task>/<dataset>/<setting>/<model>/        main.py
    best_run.txt                    index of the selected configuration
    hyperparameters_<i>/
      scores.csv
      hyperparameters.json
      model.pt
  reinit/<task>/<dataset>/<setting>/<model>/        run_reinit.py
    scores.csv                      mean and std over the runs
    run_<i>/
      scores.csv
      hyperparameters.json
      model.pt
  paper/<table>/                                    paper.reproduce
    results.csv
    <entry>/run_<i>/...
```

`<task>` is `colored_leaf_counting`, `node_prediction` or `link_prediction`;
`<setting>` is `T200_K1` ... `T600_K3` for ColoredLeafCounting and `default`
otherwise.

| File | Content |
|---|---|
| `scores.csv` of a run | one row per split (`Train`, `Validation`, `Test`): the loss and the metrics of the best epoch |
| `hyperparameters.json` | the configuration, the seed and the `--deterministic` flag |
| `model.pt` | the weights of the best epoch, for `model.load_state_dict` |
| `scores.csv` of `reinit/.../<model>` | the mean and std of every column over the runs |
| `results.csv` of a table | the mean and std of every entry next to the paper's |

## Gathering a table

The test scores of every assessed model on ColoredLeafCounting:

```python
import glob

import pandas as pd

summaries = {}
for path in glob.glob("experiments/reinit/colored_leaf_counting/*/*/*/scores.csv"):
    dataset, setting, model = path.split("/")[3:6]
    summaries[dataset, setting, model] = pd.read_csv(path, header=[0, 1], index_col=0).loc["Test"]
print(pd.DataFrame(summaries).T.sort_index())
```

Every configuration of a grid with its validation score, best first:

```python
import glob
import json

import pandas as pd

rows = []
for run in sorted(glob.glob("experiments/tuning/node_prediction/ChickenPox/default/LAMP/hyperparameters_*")):
    with open(f"{run}/hyperparameters.json") as f:
        row = pd.json_normalize(json.load(f))
    row["validation MSE"] = pd.read_csv(f"{run}/scores.csv", index_col="Set").loc["Validation", "MSE"]
    rows.append(row)
print(pd.concat(rows).sort_values("validation MSE"))
```
