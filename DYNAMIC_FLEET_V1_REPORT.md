# V1 diagnostic report

Deterministic pretrained actor rollouts use the nominal 600 W, 90% efficiency charging model, full initial batteries, identical paired seed configuration and no retraining. These are simulation diagnostics; 600 W is a nominal value, not verified hardware. Each mode uses the same initial deployments and random seeds. Differences in trajectory after a scheduling decision are expected. Target-met means at least four active aircraft. Coverage AUC is mean coverage over executed steps. Reported waits and delays use completed observations; unfinished intervals are right-censored and recorded in the CSV. Unresolved requests count both returns without a replacement and unassigned active requests; the rollout original active-pending count is retained separately. UAV local/communication energy uses the original model's feasibility/fallback rule; counts distinguish requested versus effective targets.

| Wind | Mode | Deficit | Surplus | Below steps | Target met | Coverage | Queue | Wait | Replacement delay | Return energy J | Charger use |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Low Wind | naive_threshold | 821.200 | 22.900 | 430.400 | 0.139 | 0.909 | 0.559 | 12.816 | 0.294 | 1179987.054 | 0.710 |
| Moderate Wind | naive_threshold | 655.900 | 22.000 | 413.400 | 0.173 | 0.855 | 0.375 | 9.150 | 0.179 | 1169618.754 | 0.663 |
| Strong Wind | naive_threshold | 659.900 | 14.100 | 415.600 | 0.169 | 0.922 | 0.556 | 11.178 | 0.227 | 1304575.733 | 0.799 |
| Low Wind | no_replacement | 1788.500 | 0.000 | 449.000 | 0.102 | 0.529 | 0.157 | 19.575 | NA | 189710.284 | 0.130 |
| Moderate Wind | no_replacement | 1793.400 | 0.000 | 451.000 | 0.098 | 0.507 | 0.125 | 15.575 | NA | 134663.556 | 0.124 |
| Strong Wind | no_replacement | 1814.900 | 0.000 | 463.000 | 0.074 | 0.468 | 0.145 | 18.125 | NA | 116967.164 | 0.124 |
| Low Wind | target_aware | 480.700 | 22.900 | 350.600 | 0.299 | 0.909 | 1.376 | 24.624 | 4.274 | 1511737.037 | 0.882 |
| Moderate Wind | target_aware | 145.700 | 101.800 | 117.900 | 0.764 | 0.836 | 0.568 | 10.444 | 8.769 | 1595134.467 | 0.894 |
| Strong Wind | target_aware | 404.700 | 37.100 | 330.800 | 0.338 | 0.912 | 1.331 | 23.564 | 6.536 | 1525746.753 | 0.891 |

## Paired differences

Differences are target-aware minus naive, matched by wind seed and eval seed. Wins are defined by the better direction in `paired_comparisons.csv`; no statistical significance test is performed.

- Low Wind, integrated_fleet_deficit: mean Δ -340.500, median Δ -363.000; wins/ties/losses 9/0/1 (n=10).
- Low Wind, integrated_fleet_surplus: mean Δ 0.000, median Δ 0.000; wins/ties/losses 0/10/0 (n=10).
- Low Wind, target_met_ratio: mean Δ 0.160, median Δ 0.158; wins/ties/losses 8/1/1 (n=10).
- Low Wind, final_coverage: mean Δ -0.000, median Δ 0.006; wins/ties/losses 6/0/4 (n=10).
- Moderate Wind, integrated_fleet_deficit: mean Δ -510.200, median Δ -517.000; wins/ties/losses 10/0/0 (n=10).
- Moderate Wind, integrated_fleet_surplus: mean Δ 79.800, median Δ 84.500; wins/ties/losses 0/0/10 (n=10).
- Moderate Wind, target_met_ratio: mean Δ 0.591, median Δ 0.601; wins/ties/losses 10/0/0 (n=10).
- Moderate Wind, final_coverage: mean Δ -0.019, median Δ -0.025; wins/ties/losses 2/1/7 (n=10).
- Strong Wind, integrated_fleet_deficit: mean Δ -255.200, median Δ -280.500; wins/ties/losses 10/0/0 (n=10).
- Strong Wind, integrated_fleet_surplus: mean Δ 23.000, median Δ 24.000; wins/ties/losses 0/1/9 (n=10).
- Strong Wind, target_met_ratio: mean Δ 0.170, median Δ 0.178; wins/ties/losses 9/0/1 (n=10).
- Strong Wind, final_coverage: mean Δ -0.010, median Δ -0.024; wins/ties/losses 4/0/6 (n=10).

## Interpretation boundaries

1. Compare deficit and below-target means and paired differences above; the data alone do not establish statistical significance.
2. Compare surplus and above-target steps in the CSV: overlap can be intentional handover, so surplus is not automatically an erroneous launch.
3. Coverage and deficit are reported together; more active UAVs can revisit cells or compete for SAGIN service. This diagnostic does not isolate the causal contribution of either mechanism.
4. Queue length, waiting time and charger utilization indicate charging pressure. A capacity-one bottleneck cannot be established causally without a capacity ablation.
5. Return energy and replacement delay by wind class are descriptive; only paired comparisons within a wind class use identical samples. Changes across wind classes do not establish causation.
6. Rule scheduling is a necessary baseline. These results alone do not establish that a learned scheduler is necessary or that it would outperform the rules. A learning-based V2 needs a defined objective, stronger baselines, held-out conditions and a controlled ablation.
7. `mean_replacement_delay_steps` excludes unassigned requests; recovery delays exclude ongoing gaps. Observe the corresponding counts and unresolved/censored fields before interpreting these means.
8. The offload energy split mirrors the original service feasibility checks and attributes local fallback to local computation; requested and effective target counts are separately retained. No physical hardware validation is claimed.

## Answers from these diagnostics

1. **Deficit:** mean paired Δ -368.633; wins/ties/losses 29/0/1 across 30 pairs. The paired values are descriptive; no significance test was run.

2. **Below target:** mean paired Δ -153.367; wins/ties/losses 27/1/2 across 30 pairs.

3. **Overdeployment:** mean paired Δ 34.267; wins/ties/losses 0/11/19 across 30 pairs. Surplus includes deliberate handover and is not by itself proof of an unnecessary launch.

4. **Target availability:** mean paired Δ 0.307; wins/ties/losses 27/1/2 across 30 pairs.

5. **Coverage:** mean paired Δ -0.010; wins/ties/losses 12/1/17 across 30 pairs. This comparison measures the simulated outcome, not a causal effect of fleet count alone.

6. **Repeated search and contention:** mean repeated-visit ratio is 0.788 for target-aware and 0.718 for naive. Collision and contention effects were not separately measured; causation cannot be inferred.

7. **Charging pressure:** mean charger utilization is 0.889 and mean observed queue wait is 19.544 steps for target-aware. A capacity ablation is needed before calling charging the main bottleneck.

8. **Wind:** target-aware total return energy averages 1595134.467 J under moderate wind and 1525746.753 J under strong wind; observed replacement delays are 8.769 and 6.536 steps. The comparison is descriptive and includes changed trajectories and return counts.

9. **Learning:** the paired diagnostics alone do not establish a need for or benefit from a learned controller.

10. **Rule sufficiency:** whether the rule is sufficient depends on the target operating objective; define held-out conditions and stronger rule baselines before proposing V2.

11. **Minimum fleet:** 7/30 target-aware runs reached at most one search-active UAV at least once; the V0 minimum-count failure is not eliminated.

## Run provenance

The finalized diagnostic completed 90/90 runs, with zero recorded failed attempts. An earlier 90-run attempt was intentionally stopped after 36 completed rollouts and one in-progress rollout when the target-aware no-standby handover boundary was corrected. Those preliminary files remain in `dynamic_fleet_v1_pre_handover_fix`; none are included in the final paired analysis. The corrected implementation was separately rerun through the three-run pilot and all 90 formal rollouts. A 10-step two-worker diagnostic and an identical one-worker run produced matching per-run metrics and trajectories; a subsequent `--resume` found zero pending runs.

The analyzer corrects the original rollout's `unresolved_return_requests` counter by deriving unassigned requests from event records; original metrics are retained in `raw_results_original.jsonl`, `per_run_metrics_original.csv`, and the `rollout_original_unresolved_return_requests` field. `audit.json` contains the independent 90-run consistency audit.

## Validation and artifacts

All 38 unit tests passed, including the 14 V0 tests. The frozen V0 file, GA and pretrained weights were unchanged in the V1 branch. A final SHA-256 comparison of the eight original-project baseline files matched 8/8; the final run manifest also records the checked paths and the code/weights fingerprint. The pilot completed three 100-step strong-wind rollouts before the formal 90-rollout run. No completed final rollout failed or needed a retry. See `RUN_DYNAMIC_FLEET_V1.md` for exact commands and `Our_experiment/HCSAC/data/dynamic_fleet_v1/` for the CSVs, per-run event/trajectory JSON, audit, manifest and figures. These generated artifacts remain on disk but are excluded from Git.
