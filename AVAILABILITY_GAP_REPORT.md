# Availability and search continuity: nominal diagnostic

## Scope and protocol

This is a rule/planning diagnostic, not a trained GNN comparison or joint trajectory/offloading optimization. The source environment is V1.5, source commit `d17382c123eb32fa811e78e9e159936d766e9bc3`. The physical model, fleet scheduler and charging process are unchanged; `ENV_CONTRACT.json` gives units, safety and information permissions. All three policies are passed the same current observation object. P0's frozen legacy CNN consumes its local view only; P1 and P2 process the union of currently visible local views with centralized fleet telemetry. Thus P1–P0 jointly changes spatial coordination and the use of shared current information: it does not isolate a single architectural cause. Telemetry is an explicit information assumption, not a simulated physical radio link.

P1 and P2 share candidate cells, flight-energy feasibility, assignment, deterministic grid movement and budget. P2 alone adds a fixed 30-step, model-based availability estimate to assignment costs. Estimates depend solely on current lifecycle state, energy, occupancy and FIFO order; they are not future events, and they are approximate, not guaranteed recovery times. No coefficient was tuned after the corrected confirmation results.

Nine development rollouts used previously employed scenes. Before corrected confirmation, six previously excluded winds (two each low/moderate/strong) and two scene seeds per wind were frozen in `SCENARIO_MANIFEST.json`, yielding 12 paired scenes and 36 fixed-HAPS rollouts. The predeclared trigger (P1–P0 AUC ≥ 0.01 and ≥ 8/12 positive) was met, so all 36 were also run under the same frozen, state-dependent pretrained GCN offloading policy. All cases ran 500 steps. The prior exclusion list is documented in the manifest; absolute novelty relative to every historical experiment cannot be guaranteed. Wind and terrain combinations are not independent draws for population inference; each wind occurs with two terrain/infrastructure seeds.

An earlier readiness estimator incorrectly treated 95% charge as the end of `charging` (in V1.5 charging ends at 100%, while 95% qualifies an already-standby UAV). We retained those 81 diagnostic runs and their CSVs separately; their results are excluded here. After fixing the estimate and adding a regression test, we froze different confirmation scenes before observing their results. See `results/availability_gap/pre_full_charge_fix/PROTOCOL_DEVIATION.md`. No physical dynamics, trigger values or planner weights were changed.

## Confirmation results

Each mean below has 12 matched scenes. AUC and coverage are fractions; energy is MJ. A 500-step episode may have reached a threshold earlier; unreached cases are right-censored, not assigned a success time.

| Control | Policy | Coverage AUC | Final coverage | Repeated ratio | Fleet deficit (steps) | Energy (MJ) | 90% reached | 95% reached |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed HAPS | P0 | 0.8003 | 0.8979 | 0.7501 | 553.7 | 9.401 | 6/12 | 3/12 |
| fixed HAPS | P1 | 0.8864 | 0.9996 | 0.7055 | 617.4 | 9.537 | 12/12 | 12/12 |
| fixed HAPS | P2 | 0.8689 | 0.9906 | 0.7011 | 654.8 | 9.598 | 12/12 | 11/12 |
| shared pretrained GCN | P0 | 0.7922 | 0.8838 | 0.7499 | 576.4 | 9.317 | 5/12 | 2/12 |
| shared pretrained GCN | P1 | 0.8861 | 1.0000 | 0.6926 | 674.7 | 9.538 | 12/12 | 12/12 |
| shared pretrained GCN | P2 | 0.8638 | 0.9833 | 0.6900 | 711.4 | 9.590 | 11/12 | 11/12 |

Fixed-HAPS paired mean AUC differences: **P1−P0 = +0.08612** (12 positive, 0 ties, 0 negative); **P2−P1 = −0.01752** (0/0/12). Shared-GCN corresponding values are +0.09383 (12/0/0) and −0.02225 (0/0/12). These are descriptive paired differences, not a statistical significance or generalization claim. For P2−P1, fixed-HAPS low/moderate/strong means are −0.01019/−0.01848/−0.02390; the shared-GCN means are −0.01624/−0.02843/−0.02207. Different winds were not manipulated as a causal intervention.

Under fixed HAPS, conditional mean time to 90% among runs reaching it was P0 173.5 steps (6/12), P1 116.7 (12/12), P2 131.7 (12/12). Conditional 95% times were 214.0 (3/12), 125.9 (12/12), 143.2 (11/12). Conditional energy to 90% was 3.337/2.416/2.803 MJ respectively; the conditioning differs across policies, so these are not matched-sample causal energy comparisons. Every unreached case and energy omission is retained in per-run CSV; do not average the 500-step censor time as an event time.

Mean queue length under fixed HAPS was 1.47/1.61/1.70, completed-wait mean 26.0/27.7/29.2 steps, and runs touching one search-active UAV 3/6/5, respectively. This demonstrates substantial waiting and intermittent availability loss in the nominal setup, but without a charger-capacity or fleet-size intervention cannot identify charging as the unique main cause. P1's higher deficit despite better coverage also shows fleet count alone is not search performance.

For post-charge relaunch, observed first-new-coverage delays were 86/55/77 and censored delays were 227/265/249 for P0/P1/P2. Reported completed-only mean per-run delays (8.42/4.42/2.95 steps) have heavy, policy-dependent censoring and must not be interpreted as the unconditional delay distribution. Many relaunches occur after the map is almost saturated. `early`, `middle`, `late`, and `near_saturation` JSON columns preserve the phase measurements.

The ±10-step rotation windows occupy on average 468.6/473.0/476.9 of 500 steps under fixed HAPS: ordinary windows are only 31.4/27.0/23.1 steps, predominantly early. Window-specific gain rates are preserved, but their direct comparison is confounded by time and saturation. They cannot establish that benefits or losses are uniquely caused by rotations. The shared GCN requested target counts were `[0,0,15014,0,33975]` and effective counts `[17034,0,14902,0,17053]` across the 36 runs (index order is the original five-action environment); fallback is retained. Fixed HAPS requested `[0,0,50586,0,0]`, effective `[910,0,49676,0,0]`. This is not evidence that HAPS is globally optimal.

## Information path and interpretation

On an environment-derived controlled snapshot, altering an isolated charging UAV's energy ratio from 0.2 to 0.8 or its waiting age from 2 to 8 changed explicit central telemetry, but changed neither active-node GCN nor LAMP embedding (maximum observed difference 0); the aggregate charger/queue values are broadcast and may still affect active nodes. The test uses input perturbations only, not performance scenes. See `CONTROL_INFORMATION_AUDIT.md` and `results/availability_gap/sensitivity.json`.

1. **General spatial coordination:** current-view matching materially improves AUC in these matched cases relative to the frozen local CNN. P0 also processes less centralized information, so the contrast includes both policy and information processing.
2. **Rotation-specific continuity:** the tested P2 forecast did not outperform P1; heavy censoring and overlapping windows prevent identifying a unique rotation effect. A poor forecast score is not proof that all availability-aware planning is useless.
3. **Physical energy supply:** recurrent queueing, fleet deficit and one-aircraft minima are observed; without controlled supply ablation, the main causal bottleneck remains unproven.
4. **Action permissions:** all use the existing safety/charging/return overrides. Our evidence does not isolate action authority as a bottleneck; no autonomy for return was added.
5. **Insufficient evidence:** no conclusion that a GNN, memory mechanism, ECC, extra charger or a changed reward is necessary; no real-world or joint-optimization claim.

## Reproduction and artifacts

Run `PYTHON_BIN=/opt/miniconda3/envs/SAGIN/bin/python bash scripts/run_availability_gap.sh`. Existing matching raw runs are resumed without overwrite; fingerprint mismatch fails safely. `results/availability_gap/` contains per-run, aggregate and paired CSV, sensitivity audit, regression output, and provenance. Local `outputs/availability_gap/` holds full timelines/events and is intentionally untracked. The published lightweight bundle excludes checkpoints and large replay pools.
