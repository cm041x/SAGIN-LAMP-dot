# Dynamic fleet changelog

- Added `ENV/UAVenv_DynamicFleet.py`: six stable slots, status/mask and FIFO charging, wind-aware stepped return, safe-boundary checks, event and graph masks. `step_offload` is explicitly adapted from the original implementation, not patched into it.
- Added `dynamic_fleet_smoke_test.py` and `dynamic_fleet_pretrained_test.py`: deterministic diagnostics, original actor loading, logs, summaries and figures.
- Added `tests/test_dynamic_fleet.py`, `baseline_check.py`, baseline JSONs, source SHA-256 manifest, `ORIGIN.md`, specification and run instructions.
- Added `.gitignore` for local caches and generated diagnostics. The original source, GA and model weights in this copy are not modified. The original `UAVEnv` still operates separately, and its cwd-relative wind-file behavior remains unchanged.
- A project-local Python dependency experiment was removed after discovering the existing conda environment; the retained `baseline_check_python311.json` records that earlier probe.

## V1

- Preserved V0 and added `fleet_scheduler.py` and `UAVenv_DynamicFleetV1.py`, including no-replacement, V0-naive and target-aware scheduling, deterministic eligible reserve selection, explicit binding lifecycle, and nominal W/efficiency charging.
- Added V1 unit tests, paired-seed rollout/analysis tools and V1 specification and run documentation. Original policies, GA and the V0 environment are unchanged.

## V1.5

- Added a V1-derived environment and optional deterministic responsibility/waypoint rule. V0, V1, GA and actor weights remain unchanged.
- Added explicit per-step visit, movement-conflict, proximity and SAGIN-target proxy instrumentation, waypoint lifecycle records, pilot/formal runners, auditing, paired analysis and tests.
- The V1.5 shared GA deployment differs from the original V1 corner deployment; comparisons are within the new paired V1.5 runs. Two metrics that V1 cannot define reliably are marked unavailable.

## Dynamic HCSAC baselines — in progress

- Added an independent masked discrete SAC and cloned CPU replay records for the original CNN/GCN architectures; no original SAC or environment code is changed.
- Added disjoint split and paired episode manifests, actor initialization checks, resumable complete checkpoints, pilot training/evaluation and analysis/audit entry points. Formal training/evaluation completion and acceptance remain pending until all requested runs are independently audited.
# LAMP prototype isolation

- The new independent project snapshots the source worktree at V1.5 HEAD,
  including its uncommitted Dynamic HCSAC training source. No old project
  history is imported. The old long-running job was stopped at the user's
  direction; see `OLD_TRAINING_STOP_RECORD.md`.
- `external/LAMP/` is the pinned official MIT-licensed source snapshot.
  `lamp_method/` contains the PyTorch dense-reference-checked propagation
  adapter, lifecycle context, candidate event-conditioned team critic, raw
  sequence replay, collector and bounded pilot entry points.
- The isolated copy's `tests/test_dynamic_hcsac_training.py` now verifies the
  V1.5 file against the snapshot SHA-256 instead of requiring a commit object
  from the old repository; its CPU thread cap is two. The original environment,
  GA inputs, pretrained weights and physics are not edited.
- The prototype uses the frozen original CNN and a newly trained offloading
  branch. It does not implement a learned fleet scheduler or assert superiority.
