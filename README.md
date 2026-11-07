# SAGIN-LAMP — portable research snapshot

This repository contains the multi-UAV SAGIN search environment, fixed GA deployments, HCSAC baselines and bounded LAMP offloading and flight-residual pilots. It is an experimental snapshot, not a validated production system or a full-paper reproduction. Source repository commit: `45093e616ad6f99f5eff8122f1404caefe4cc9e9` (this is distinct from this repository's publication commit).

## Setup

The original project recommends Python 3.12.5. From the repository root, create an isolated environment and install the pinned dependencies:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch and torch-geometric wheels depend on the platform; consult their installation instructions if pip cannot resolve them. The upstream LAMP project under `external/LAMP/` has its own dependencies and MIT license; the SAGIN adapter does not require running the upstream project separately. The legacy encrypted environment includes Linux x86-64, macOS universal and Windows x86-64 PyArmor runtimes; compatibility with other platforms or Python versions is not established.

## Files and entry points

- Original frozen flight CNN: `Our_experiment/HCSAC/data/sac_model_fly.pt`; original offloading GCN: `Our_experiment/HCSAC/data/sac_model_offload.pt`.
- Wind: `Our_experiment/HCSAC/wind.json` and `Our_experiment/GA/wind.json`; wind classifications: `Our_experiment/HCSAC/Our_experiment/HCSAC/data/wind_seed_catalog_5000.csv` and `wind_seed_classes_5000.json` in the same directory.
- Fixed GA deployments: `Our_experiment/GA/data/ga_best_deployment_random10_w{11,23,4800}_g10_i999999_pool0_200.json`.
- Pilot reports and results: `LAMP_PROTOTYPE_REPORT.md`, `FLIGHT_RESIDUAL_PILOT_REPORT.md`, `outputs/lamp_prototype/`, and `outputs/lamp_flight_residual_pilot/`. Only completed summary CSVs, configuration and reports are included; intermediate checkpoints and trajectories are excluded.
- Tests (from the repository root):

```bash
PYTHONPATH=. SDL_VIDEODRIVER=dummy MPLBACKEND=Agg python -m unittest discover -s tests -p 'test_lamp_prototype.py' -v
PYTHONPATH=. SDL_VIDEODRIVER=dummy MPLBACKEND=Agg python -m unittest discover -s tests -p 'test_flight_residual.py' -v
```

- Existing full experiment entry points (these start training; set the interpreter explicitly):

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_lamp_prototype.sh
PYTHON_BIN=.venv/bin/python bash scripts/run_lamp_flight_residual_pilot.sh
```

See `RUN_LAMP_PROTOTYPE.md` and `RUN_FLIGHT_RESIDUAL.md` for stages and budgets; their original local paths are historical and should not be used. The published summaries do not contain the checkpoints needed to resume the original runs.

## Known limitations

The flight pilot used one training seed and nine evaluation scenarios; all LAMP variants had lower mean final coverage than the frozen CNN, and the GCN residual performed much worse. Large LAMP hidden norms, potentially stale task features and the inherited update ordering limit interpretation. See `FLIGHT_RESIDUAL_PILOT_REPORT.md` and `ENVIRONMENT_SEMANTICS_AUDIT.md`. No claims of convergence or real-world performance follow from these pilots.

