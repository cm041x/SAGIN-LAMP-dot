# Traffic forecasting

The traffic experiments (METR-LA, PeMS-Bay) are a separate
[tsl](https://torch-spatiotemporal.readthedocs.io/) pipeline in
`traffic_forecasting/`, which reuses the LAMP convolution of `src/`. Install
the extra dependencies with

```bash
uv sync --extra cu121 --extra traffic
```

and run

```bash
cd traffic_forecasting
python run_traffic_experiment.py --dataset la
python run_traffic_experiment.py --dataset bay
```

`--dataset` loads `la.yaml` or `bay.yaml`. Any other argument is a Hydra
override of that file, e.g. `device=0 seed=2 epochs=100`. tsl downloads the
datasets on first use.
