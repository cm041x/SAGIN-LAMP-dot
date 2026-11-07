#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/SAGIN/bin/python}"
"$PYTHON_BIN" -m Our_experiment.HCSAC.lamp_method.run --stage test
"$PYTHON_BIN" -m Our_experiment.HCSAC.lamp_method.run --stage smoke
"$PYTHON_BIN" -m Our_experiment.HCSAC.lamp_method.run --stage math-check
"$PYTHON_BIN" -m Our_experiment.HCSAC.lamp_method.run --stage pilot --resume
"$PYTHON_BIN" scripts/validate_lamp_pilot.py
"$PYTHON_BIN" -m Our_experiment.HCSAC.lamp_method.run --stage evaluate --resume
"$PYTHON_BIN" -m Our_experiment.HCSAC.lamp_method.run --stage report
"$PYTHON_BIN" scripts/plot_lamp_status.py
"$PYTHON_BIN" scripts/audit_lamp_pilot.py
"$PYTHON_BIN" scripts/audit_lifecycle_weights.py
"$PYTHON_BIN" scripts/diagnose_lamp_actions.py
"$PYTHON_BIN" scripts/diagnose_effective_offloads.py
