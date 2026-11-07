# Flight-residual pilot

From this project root, run:

```bash
bash scripts/run_lamp_flight_residual_pilot.sh
```

The command runs the environment and original LAMP regression suites, checks
zero-residual CNN parity, runs five 500-step integration smokes, trains four
models within the fixed 20-episode budget, evaluates the last checkpoints and
the frozen CNN, and writes a report. It uses the existing SAGIN conda Python
by default and CPU threads are limited to two. `PYTHON_BIN` may select another
compatible interpreter without modifying that environment.

Artifacts are saved under `outputs/lamp_flight_residual_pilot/`; the original
`outputs/lamp_prototype/` stays untouched. Re-running the command resumes only
checkpoints whose configuration, code fingerprint, inputs and model hashes
match. Its 200-step training and 500-step evaluation are deliberately different
and do not demonstrate convergence. The fixed HAPS request isolates flight
decisions and does not claim global offloading optimality. See
`FLIGHT_RESIDUAL_SPEC.md` for the observation and reward semantics.
