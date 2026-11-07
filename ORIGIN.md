# Source and isolation

- Source: `/Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-DynamicFleet-Charging`
- Destination: `/Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-LAMP`
- Source HEAD: `e4a5c58cef474751a414de60ecb7fc61593beabf`
- Copy time: 2026-11-05T20:27:27+08:00
- Source worktree was dirty; copied current files, including uncommitted training source, not just HEAD.
- Old project and original paper project are read-only during LAMP implementation. Original pretrained weights and GA inputs are preserved.
- The authorized old HCSAC training was stopped via SIGINT after checkpoint verification; it will not be restarted for this prototype.

## Source status at copy

```
 M .gitignore
 M CHANGELOG_DYNAMIC_FLEET.md
 M README.md
?? DYNAMIC_HCSAC_BASELINE_REPORT.md
?? DYNAMIC_HCSAC_BASELINE_SPEC.md
?? Our_experiment/HCSAC/analyze_dynamic_hcsac.py
?? Our_experiment/HCSAC/audit_dynamic_hcsac.py
?? Our_experiment/HCSAC/evaluate_dynamic_hcsac.py
?? Our_experiment/HCSAC/train_dynamic_hcsac.py
?? Our_experiment/HCSAC/training/
?? RUN_DYNAMIC_HCSAC.md
?? tests/test_dynamic_hcsac_training.py
```

## Untracked source and documentation copied

```
DYNAMIC_HCSAC_BASELINE_REPORT.md
DYNAMIC_HCSAC_BASELINE_SPEC.md
Our_experiment/HCSAC/analyze_dynamic_hcsac.py
Our_experiment/HCSAC/audit_dynamic_hcsac.py
Our_experiment/HCSAC/evaluate_dynamic_hcsac.py
Our_experiment/HCSAC/train_dynamic_hcsac.py
Our_experiment/HCSAC/training/__init__.py
Our_experiment/HCSAC/training/checkpointing.py
Our_experiment/HCSAC/training/dynamic_hcsac_trainer.py
Our_experiment/HCSAC/training/masked_discrete_sac.py
Our_experiment/HCSAC/training/replay_buffers.py
Our_experiment/HCSAC/training/training_split.py
RUN_DYNAMIC_HCSAC.md
tests/test_dynamic_hcsac_training.py
```

Complete source and fixed input SHA-256 hashes are in `SOURCE_SNAPSHOT_SHA256.json`.

## Key source and fixed input SHA-256

- `Our_experiment/HCSAC/ENV/UAVenv_SAC_Original.py`: `12ec3bb6b171442d3963c0266936d79a11cc51fcf88d8e85d738d54ca25d7419`
- `Our_experiment/HCSAC/ENV/UAVenv_DynamicFleetV15.py`: `b40d0bb3e5f5e4b6b991b675de34d853965b4815d2c1c0872f9241222640e0ce`
- `Our_experiment/HCSAC/training/dynamic_hcsac_trainer.py`: `99cf48f7f29e539b80cf10b80cefbf22d6a94a7c65f827482427eb13de452e6d`
- `Our_experiment/HCSAC/data/sac_model_fly.pt`: `1f05d4b0e2877b6f42fa12251b80d4d2b2944a90d857aeb9789177fd2106a996`
- `Our_experiment/HCSAC/data/sac_model_offload.pt`: `58ee602faecd02bd820cc25af5343cfea30f34d9c417823f72047a88c07a4832`
- `Our_experiment/HCSAC/wind.json`: `9d7f06c8cc32d1b4c9903240ffa2466fcc1c57a1eca38d88a12b7a1668039331`
- `Our_experiment/HCSAC/Our_experiment/HCSAC/data/wind_seed_classes_5000.json`: `28c236da73aa98e2d308667fcff0bc383e7059af5c63f2ae5bad5a0585367e23`
- `Our_experiment/GA/data/ga_best_deployment_random10_w23_g10_i999999_pool0_200.json`: `9ba11b2d07343f378c9fd5dd020984d651035c9ad49ac088b0ee28de4bbaec8a`
