# Run dynamic fleet V0

Use the existing conda environment `SAGIN`; it has Python 3.12, PyTorch 2.4.1 and PyG 2.6.1. The commands below do not install packages or modify that environment. For a different machine create/activate a compatible conda environment and install the versions in `requirements.txt`; do not change existing weights.

```sh
cd /Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-DynamicFleet-Charging
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.mplconfig" OMP_NUM_THREADS=4
PY=/opt/miniconda3/envs/SAGIN/bin/python
$PY baseline_check.py
$PY -m unittest discover -s tests -p 'test_dynamic_fleet*.py' -v
$PY Our_experiment/HCSAC/dynamic_fleet_smoke_test.py --seed 42 --max-steps 200 --output-dir Our_experiment/HCSAC/data/dynamic_fleet_smoke --controller heuristic
$PY Our_experiment/HCSAC/dynamic_fleet_pretrained_test.py --seed 42 --max-steps 200 --output-dir Our_experiment/HCSAC/data/dynamic_fleet_pretrained
```

Both diagnostic runners write `events.jsonl`, `snapshots.json`, `summary.json`, fleet/count/energy/uncertainty/status PNGs to their output directory. The pretrained test uses the original checkpoint actor weights, greedy argmax actions and no training. Failures produce a nonzero exit; the pretrained runner saves a traceback in its output directory. `baseline_check.py` uses the original environment from its required HCSAC working directory and writes `baseline_check.json` at project root.

A missing wind file usually means the original `UAVEnv` was instantiated from the wrong working directory: the provided runners change to `Our_experiment/HCSAC`. Import errors mean the wrong Python was selected; check `$PY -c 'import torch, torch_geometric'`. Matplotlib should use Agg and a writable `MPLCONFIGDIR`. Use separate output directories to retain distinct runs; existing files in an output directory are overwritten by a new run.
