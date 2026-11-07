# Run dynamic HCSAC baselines

Use the existing `SAGIN` conda environment. GPU `auto` selects CUDA only; this machine's CUDA-unavailable path uses CPU, not MPS. No dependencies or original weights need modification.

```sh
cd /Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-DynamicFleet-Charging
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.mplconfig" OMP_NUM_THREADS=4
PY=/opt/miniconda3/envs/SAGIN/bin/python
$PY -m unittest discover -s tests -p 'test_dynamic*.py' -v
$PY Our_experiment/HCSAC/train_dynamic_hcsac.py --modes scratch --train-seeds 3001 --episodes 10 --max-steps 50 --warmup-steps 32 --batch-size 16 --validation-every 5 --output-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/pilot/scratch --device auto --resume
$PY Our_experiment/HCSAC/train_dynamic_hcsac.py --modes finetune --train-seeds 3001 --episodes 10 --max-steps 50 --warmup-steps 32 --batch-size 16 --validation-every 5 --output-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/pilot/finetune --device auto --resume
$PY Our_experiment/HCSAC/train_dynamic_hcsac.py --modes scratch --train-seeds 3001 --episodes 15 --max-steps 50 --warmup-steps 32 --batch-size 16 --validation-every 5 --output-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/pilot/scratch --device auto --resume
$PY Our_experiment/HCSAC/evaluate_dynamic_hcsac.py --training-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/pilot --scratch-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/pilot/scratch --finetune-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/pilot/finetune --include-zero-shot --modes scratch,finetune --train-seeds 3001 --wind-seeds 4800 --eval-seeds 42 --max-steps 100 --output-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/pilot/evaluation --workers 1 --resume
$PY Our_experiment/HCSAC/train_dynamic_hcsac.py --modes scratch,finetune --train-seeds 3001,3002,3003 --episodes 1000 --max-steps 500 --warmup-steps 2000 --batch-size 128 --validation-every 50 --checkpoint-every 50 --output-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/training --device auto --resume
$PY Our_experiment/HCSAC/evaluate_dynamic_hcsac.py --training-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/training --include-zero-shot --modes scratch,finetune --train-seeds 3001,3002,3003 --wind-seeds 11,23,4800 --eval-seeds 42,43,44,45,46,47,48,49,50,51 --max-steps 500 --output-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/evaluation --workers 1 --resume
$PY Our_experiment/HCSAC/audit_dynamic_hcsac.py --training-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/training --evaluation-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/evaluation
$PY Our_experiment/HCSAC/analyze_dynamic_hcsac.py --training-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/training --evaluation-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/evaluation --output-dir Our_experiment/HCSAC/data/dynamic_hcsac_baselines/analysis
```

Model/output directories and checkpoint data are ignored by Git. The trainer uses complete episode-boundary checkpoints, and rejects a changed code/configuration or incompatible episode manifest on resume. Evaluation currently supports `--workers 1` only. A completed pilot is not a trained formal baseline. On this CPU, the measured batch-128 update-only lower bound for the requested six full runs was approximately 171 hours; environment steps, validation and checkpoint I/O add time.
