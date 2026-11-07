# Installation

The environment is managed with [uv](https://docs.astral.sh/uv/) (install it
with `curl -LsSf https://astral.sh/uv/install.sh | sh`). Everything is pinned in
`pyproject.toml` and `uv.lock`; you only choose the build of PyTorch that
matches your machine:

```bash
git clone https://github.com/FabriDeCastelli/LAMP.git
cd LAMP
uv sync --extra cu121      # pick one, see the table below
source .venv/bin/activate
```

| Your machine | Extra |
|---|---|
| NVIDIA GPU, driver for CUDA 12.1 or newer (`nvidia-smi` shows "CUDA Version: 12.x") | `--extra cu121` |
| NVIDIA GPU, driver for CUDA 11.8 | `--extra cu118` |
| No GPU | `--extra cpu` |

All commands in these docs assume the environment is activated. Without
activating it, prefix them with `uv run --extra <your extra>`, e.g.
`uv run --extra cpu python -m paper.reproduce paper/clc.yaml`.

The lockfile targets Linux (the platform with prebuilt PyTorch Geometric
extension wheels for this PyTorch version).

The traffic experiments need one more extra, see [traffic.md](traffic.md).

## Checking the installation

```bash
pytest tests
```

trains LAMP for a few epochs on ColoredLeafCounting and ChickenPox on CPU and
checks that the losses match, bit for bit, those of the code that produced the
paper. It takes a few minutes and downloads the datasets it needs.
