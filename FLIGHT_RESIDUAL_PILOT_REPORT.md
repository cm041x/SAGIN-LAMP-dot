# Flight-residual pilot results

This is a bounded engineering pilot, not a convergence, novelty, or generalization claim. The fixed-HAPS offloading controller isolates flight decisions; it is not a claim that HAPS is globally optimal. The V1.5 physical environment, rewards, deployment and charging rules were not changed. `FLIGHT_RESIDUAL_SPEC.md` documents observation and transition semantics.

## Protocol and checks

The frozen reference and four trained variants share the original frozen CNN, local features, lifecycle context, physical graph and HAPS request with the environment's local fallback. Each trained variant used train seed 3001, 20 episodes of 200 steps, two warmup episodes and 360 updates. Last checkpoints were assessed on wind seeds 11, 23, 4800 and evaluation seeds 42, 43, 44: nine paired 500-step rollouts per policy. This training/evaluation horizon difference is material. Test scenarios were not used to select checkpoints.

The test entry point passed 97 dynamic-environment tests, 28 original LAMP tests and 12 new flight tests. All four zero-initialized residual policies match the original CNN to maximum absolute probability error 1.01e-8. Five 500-step zero-head smoke runs produced the same trajectory, with 29 return-rule overrides each. ECC's two event-conditioned critic heads receive updates; no inactive offloads or energy-exhaustion events occurred in the 45 evaluation runs. The pretrained CNN's file and parameter hashes remained unchanged. The original offloading prototype's 126 files and source-project snapshots passed integrity checks.

## Mean of nine evaluation runs

| Policy | Final coverage | Coverage AUC | Repeated visit ratio | UAV energy (J) | Return energy (J) | Integrated fleet deficit | Action-change ratio | Trajectories changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen CNN | 0.9236 | 0.8187 | 0.7803 | 9,205,269 | 1,582,977 | 338.7 | 0.0000 | 0/9 |
| GCN residual | 0.3500 | 0.3339 | 0.9204 | 9,154,765 | 1,719,396 | 284.8 | 0.9374 | 9/9 |
| Plain LAMP residual | 0.8972 | 0.7960 | 0.7923 | 9,278,227 | 1,488,926 | 288.0 | 0.0519 | 9/9 |
| Lifecycle LAMP residual | 0.8986 | 0.7988 | 0.7754 | 9,342,062 | 1,592,449 | 393.6 | 0.0998 | 9/9 |
| Lifecycle LAMP + ECC residual | 0.9069 | 0.8022 | 0.7750 | 9,286,864 | 1,654,917 | 389.8 | 0.1080 | 9/9 |

Action-change ratios in the table average per-run ratios; each compares the proposed argmax against the frozen CNN on the same encountered state. They are not identical to comparing different trajectories timestep by timestep. Mean paired final-coverage differences versus the frozen CNN were -0.5736, -0.0264, -0.0250 and -0.0167 respectively. The corresponding better/same/worse counts over the nine matched scenarios were 0/0/9, 1/0/8, 2/0/7 and 2/0/7. No significance claim is made.

The four actors, both critics, graph encoders and residual heads changed parameters. The lifecycle models' mean conditioner gradient norms were 1.22e-7 and 2.08e-7. ECC training event cross-entropy was 0.686; in the nine held-out evaluation runs, its mean event cross-entropy was 0.626, Brier score 0.278, and accuracy 0.846, equal to the per-run majority-class reference of 0.846. This is no evidence of a useful event predictor. Action changes establish that the path head has an effect; they do not establish a benefit. All three LAMP variants changed trajectories in all nine cases yet had lower mean coverage than the reference. GCN residual changed almost all actions and had much lower coverage, a negative result to preserve rather than tune away.

## Limits and reproducibility

Only one training seed and nine evaluation scenarios were used. The task feature inherited from the existing environment may remain stale, and uncertainty is updated before offloading in that environment. Neither was silently altered. Mean evaluation LAMP hidden norms were approximately 22,394–25,250, while the GCN mean was about 24; these large LAMP norms are a numerical risk, not a stability guarantee. The short pilot cannot isolate causality or establish convergence. All methods requested HAPS; feasible effective actions and local fallback counts are recorded separately. The earlier offload-only results choosing HAPS do not establish HAPS optimality in other states.

Run `bash scripts/run_lamp_flight_residual_pilot.sh` from the project root. The ignored `outputs/lamp_flight_residual_pilot/` directory contains checkpoints, per-run trajectories/actions, training metrics, paired inputs, CSV summaries and the machine-readable audit. Existing `outputs/lamp_prototype/` is left intact. Reproduction on another machine can differ numerically and requires the same inputs and compatible Python/PyTorch environment.
