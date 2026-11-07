# Run V1.5

```sh
cd /Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-DynamicFleet-Charging
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.mplconfig" OMP_NUM_THREADS=4
PY=/opt/miniconda3/envs/SAGIN/bin/python
$PY -m unittest discover -s tests -p 'test_dynamic_fleet*.py' -v
$PY Our_experiment/HCSAC/dynamic_fleet_v15_capacity_experiment.py --target-search-uavs 3,4 --charging-capacities 1,2 --wind-seeds 4800 --eval-seeds 42 --max-steps 150 --charging-power-w 600 --charging-efficiency 0.9 --output-dir Our_experiment/HCSAC/data/dynamic_fleet_v15_capacity_pilot --workers 1 --resume
$PY Our_experiment/HCSAC/dynamic_fleet_v15_handover_experiment.py --methods naive_threshold,target_aware,target_aware_handover --wind-seeds 4800 --eval-seeds 42 --max-steps 150 --charging-power-w 600 --charging-capacity 1 --target-search-uavs 4 --output-dir Our_experiment/HCSAC/data/dynamic_fleet_v15_handover_pilot --workers 1 --resume
$PY Our_experiment/HCSAC/dynamic_fleet_v15_capacity_experiment.py --target-search-uavs 3,4 --charging-capacities 1,2 --wind-seeds 11,23,4800 --eval-seeds 42,43,44,45,46,47,48,49,50,51 --max-steps 500 --charging-power-w 600 --charging-efficiency 0.9 --output-dir Our_experiment/HCSAC/data/dynamic_fleet_v15_capacity --workers 1 --resume
$PY Our_experiment/HCSAC/dynamic_fleet_v15_handover_experiment.py --methods naive_threshold,target_aware,target_aware_handover --wind-seeds 11,23,4800 --eval-seeds 42,43,44,45,46,47,48,49,50,51 --max-steps 500 --charging-power-w 600 --charging-capacity 1 --target-search-uavs 4 --output-dir Our_experiment/HCSAC/data/dynamic_fleet_v15_handover --workers 1 --resume
$PY Our_experiment/HCSAC/audit_dynamic_fleet_v15.py --capacity-dir Our_experiment/HCSAC/data/dynamic_fleet_v15_capacity --handover-dir Our_experiment/HCSAC/data/dynamic_fleet_v15_handover
$PY Our_experiment/HCSAC/analyze_dynamic_fleet_v15_capacity.py --input-dir Our_experiment/HCSAC/data/dynamic_fleet_v15_capacity
$PY Our_experiment/HCSAC/analyze_dynamic_fleet_v15_handover.py --input-dir Our_experiment/HCSAC/data/dynamic_fleet_v15_handover
```

A run gets its own directory under `runs/` and is published when all its outputs finish. `--resume` skips completed runs under an identical full configuration and code/data fingerprint; use a new directory after code changes. Failure tracebacks are appended to `failed_runs.jsonl`. Outputs are ignored by Git and preserved on disk. V0 and V1 retain their own run commands and datasets.

The analyzers preserve original raw files and relabel the simulator's partial occupied-cell metric as `sequential_blocked_move_attempts`; the requested complete simultaneous same-cell attempt count remains unavailable. Run the audit again after analysis to verify normalized summaries/CSV.
