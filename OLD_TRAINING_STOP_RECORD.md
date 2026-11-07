# Old HCSAC training stop record

- Stopped the specifically verified `train_dynamic_hcsac.py --modes scratch,finetune --train-seeds 3001,3002,3003 --episodes 1000` process using SIGINT after verifying cwd and command.
- Latest checkpoint after exit loaded successfully: scratch seed 3001, episode 685, global step 342500, best episode 150. CNN/GCN replay and Python/NumPy/Torch RNG states are present.
- Checkpoint: `/Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-DynamicFleet-Charging/Our_experiment/HCSAC/data/dynamic_hcsac_baselines/training/scratch/seed_3001/last_checkpoint.pt`.
- The process may have lost unsaved partial-episode in-memory progress after the checkpoint. The existing best, last, logs, configuration and original weights were not deleted or overwritten.
- This is a training diagnostic; a temperature/entropy configuration issue was identified. It is not a validated formal baseline.
