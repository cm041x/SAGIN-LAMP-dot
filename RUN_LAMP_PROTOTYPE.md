# Running the LAMP prototype

```bash
cd /Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-LAMP
bash scripts/run_lamp_prototype.sh
```

The script uses `/opt/miniconda3/envs/SAGIN/bin/python` by default. Override
with `PYTHON_BIN=/path/to/python` if needed. It runs CPU-only with two threads,
does not install packages, and stores all artifacts under
`outputs/lamp_prototype/`. The test, parity, smoke and math gates run before
the sequential 20-episode pilots. Training and evaluation use `--resume`, with
episode-boundary model, optimizer, raw replay and RNG checkpoints. An
interrupted episode is rerun from the last checkpoint. The script does not
start any 1000-episode HCSAC run.

Individual stages use `python -m Our_experiment.HCSAC.lamp_method.run --stage`
with `test`, `smoke`, `math-check`, `pilot --resume`, `evaluate --resume`, or
`report`. Set `PYTHONPATH` to this project root when invoking stages directly.
