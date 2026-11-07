#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/SAGIN/bin/python}"
"$PYTHON_BIN" -m Our_experiment.HCSAC.flight_residual.run --stage test
"$PYTHON_BIN" -m Our_experiment.HCSAC.flight_residual.run --stage initial-check
"$PYTHON_BIN" scripts/check_flight_residual_smoke.py
"$PYTHON_BIN" -m Our_experiment.HCSAC.flight_residual.run --stage pilot --resume
"$PYTHON_BIN" -m Our_experiment.HCSAC.flight_residual.run --stage evaluate --resume
"$PYTHON_BIN" -m Our_experiment.HCSAC.flight_residual.run --stage report
