# Run V1

Use the existing `SAGIN` conda environment; no installation or retraining is needed. These commands run on CPU and cap total PyTorch threads at four. Paths below are relative to this project.

```sh
cd /Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-DynamicFleet-Charging
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.mplconfig" OMP_NUM_THREADS=4
PY=/opt/miniconda3/envs/SAGIN/bin/python
$PY -m unittest discover -s tests -p 'test_dynamic_fleet*.py' -v
$PY Our_experiment/HCSAC/dynamic_fleet_v1_experiment.py --methods no_replacement,naive_threshold,target_aware --wind-seeds 4800 --eval-seeds 42 --max-steps 100 --charging-power-w 600 --charging-capacity 1 --target-search-uavs 4 --output-dir Our_experiment/HCSAC/data/dynamic_fleet_v1_pilot --resume --workers 1
$PY Our_experiment/HCSAC/dynamic_fleet_v1_experiment.py --methods no_replacement,naive_threshold,target_aware --wind-seeds 11,23,4800 --eval-seeds 42,43,44,45,46,47,48,49,50,51 --max-steps 500 --charging-power-w 600 --charging-capacity 1 --target-search-uavs 4 --output-dir Our_experiment/HCSAC/data/dynamic_fleet_v1 --resume --workers 1
$PY Our_experiment/HCSAC/audit_dynamic_fleet_v1.py --input-dir Our_experiment/HCSAC/data/dynamic_fleet_v1
$PY Our_experiment/HCSAC/analyze_dynamic_fleet_v1.py --input-dir Our_experiment/HCSAC/data/dynamic_fleet_v1
```

Each run uses `wind_seed` for wind and `eval_seed` for terrain, infrastructure and controller RNG; actor selection uses argmax. Run IDs identify mode, wind and eval seed. Per-run JSON files hold metrics, events and trajectory and are written atomically; `--resume` skips completed files only for exactly the same configuration and code/weights fingerprint. Failures are appended to `failed_runs.jsonl` and can be retried using `--resume`. A configuration or code change needs a fresh output directory to avoid mixing incompatible runs. `--workers` supports 1–4 processes, each with a private environment, RNG, and disjoint run file; the total PyTorch thread budget is four.

The analyzer reads persisted CSV/run records to write aggregation, paired comparisons, trajectory CSV and figures. Output directories are ignored by Git but preserved on disk. Existing V0 smoke tests remain available through `RUN_DYNAMIC_FLEET.md`.

The analysis derives unresolved requests from per-run events: returns without an assigned replacement plus still-active unassigned requests. The initial rollout counter only captured active pending requests; the analyzer preserves it as `rollout_original_unresolved_return_requests`, and retains the initial raw CSV and JSONL as `*_original` files. Run the analyzer after resuming to refresh the derived tables.
