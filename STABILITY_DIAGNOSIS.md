# Flight residual stability diagnosis

## Read-only diagnosis (before new training)

The old checkpoints were replayed without modification on the fixed validation scenario `(wind, terrain, infrastructure) = (1180, 900000, 900000)`, excluded from training and formal test. In steps 101–500 the old GCN actor always selected direction 6. In the last 100 steps, mean residual range was 19.48, mean KL(new || CNN) was 13.07, and 276 attempted moves were blocked. Its residual vector was nearly constant while local input RMS varied. On those states the twin-min critic's enumerated eight-action maximum was direction 4 in all 300 checked steps, with mean Q span 0.00951, rather than direction 6. Thus the observed policy collapse is associated with a saturated near-constant head and an unconstrained residual overpowering the base policy; the snapshot critic ranking does not by itself explain the historical training trajectory, so the unique training cause remains unproven.

On the identical 500-step frozen-CNN trajectory, untrained plain LAMP with xi=1 reached hidden norm 22,845.98; actor and critic first-layer preactivations exceeded abs(3) in 99.5% and 97.5% of entries. With the same weights and inputs but xi=2, norm was 60.03 and corresponding fractions were 14.3% and 2.2%. This is a practical diagnostic threshold, not a stability theorem. At initialization, injection RMS was already nontrivial; recurrent growth and readout saturation therefore do not require training. Trained xi=1 yielded hidden norm 24,035.56 and 100% / 98.1% saturation at step 500. The xi=2 replay is a controlled diagnostic, not proof of long-horizon stability or retained memory quality.

| Policy | Final coverage | Coverage AUC | Team return | Discounted return | Coverage gain | Block penalty | Search flight energy penalty | Offload reward | Return flight energy J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen CNN | .9400 | .8368 | -39.3615 | -2.7636 | 372 | 31.35 | 14.8336 | -0.6179 | 2,151,494 |
| Old GCN | .2900 | .2812 | -78.7707 | -13.5441 | 112 | 63.45 | 15.4825 | -2.0781 | 1,526,178 |
| Old plain LAMP | .9375 | .8167 | -34.7695 | -3.1910 | 371 | 25.90 | 15.7044 | -0.5851 | 1,624,513 |

The environment also adds a 0.01 first-visit bonus and a coverage-gain term of gain/100, independently accounted for in the saved data. Return-flight energy is consumed but the returning branch produces zero flying reward, so it is not included as a return-flight energy penalty. This fixed-scenario example shows lower coverage with better *undiscounted* return for old LAMP but worse discounted return; it does not establish a general causal tradeoff. The GCN in this scenario has both lower coverage and worse returns.

## Controlled correction

The new implementation is separate from the previous experiment. It retains the original frozen CNN and environment, uses shared node-wise LayerNorm for local features and independent graph-readout LayerNorm, and retains raw recurrent hidden separately. The LAMP pilot uses xi=2. Bounded residual uses `c*tanh(raw)` with `c=1`; unbounded mode remains selectable. On a fixed training probe at steps 1, 6, 11 and 16, an actor update with mean KL(after || before) > 0.01 is rolled back together with the optimizer state, retried up to three times at successively halved per-update learning rates, and skipped after failure. This does not guarantee monotonically improving reward or coverage.

Before actor training each new model uses separately collected frozen-CNN training trajectories for critic warm-start; another disjoint training trajectory is held out to measure TD and return-prediction error. No previous optimizer is loaded. The training horizon and validation horizon are both 500 steps. Collapse stopping is fixed in advance: a direction must account for at least 90% of active decisions in the last 100 steps **and** final coverage must be at least 0.20 below the paired CNN, in at least two of the three validation scenarios. This rule records a behavioral and performance failure; it is not a hyperparameter search.

Machine-readable step-level behavior, exact return decomposition, per-node hidden RMS, propagation changes and Q arrays are under `outputs/flight_residual_stability/diagnostics/`. The short-run results and known limitations are appended after the experiment completes.

## Completed short run and validation

Both corrected policies completed the specified 10 episodes and 200 actor-stage updates. Each also received the predeclared 20 critic-only warm-start updates; no old optimizer or team critic was reused. The three validation scenarios were preselected from the existing validation split, and were run at episodes 0, 2, 4, 6, 8 and 10. The 500-step frozen-CNN reference is unchanged. The table reports the mean of those three scenarios at the last checkpoint; it is not a test-set-selected checkpoint.

| Policy | Final coverage | Coverage AUC | Team return | Discounted team return | Same-state action change | KL(new || CNN) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen CNN | .9033 | .8050 | -33.808 | -2.514 | 0 | 0 |
| Stabilized GCN | .9125 | .8143 | -35.276 | -2.889 | .0509 | .0433 |
| Stabilized plain LAMP | .8983 | .7975 | -31.624 | -3.031 | .0920 | .0533 |

GCN's three final coverages were .9175, .8725 and .9475; LAMP's were .9075, .8500 and .9375. The corresponding frozen CNN values were .9400, .8425 and .9275. Neither model met the predeclared two-part collapse-and-performance stopping rule. All 200 guarded actor updates in each group were accepted without a retry; maximum single-update fixed-probe KL was 1.04e-6 for GCN and 7.84e-7 for LAMP. The bound limits each residual component to [-1,1], hence its max-minus-min is at most 2; mean ranges on final validation scenes were already approximately 1.95–1.98, close to the limit. A low per-update KL is not a bound on accumulated drift from CNN: final mean KL(new || CNN) was .0433/.0533.

At final validation, the LAMP raw hidden RMS was .295–.360 across scenarios and its critic fusion saturation was effectively zero. Its actor first-layer saturation was .117–.311; GCN's was .132–.469. The input and recurrent scale change therefore reduced the observed LAMP accumulation and critic saturation, but did not eliminate actor saturation. No ablation separates the contribution of local LayerNorm, readout LayerNorm, xi and the residual bound.

## One independent test, after freezing the configuration

Each policy was run once for 500 steps on each of wind seeds 11, 23 and 4800, all with evaluation seed 42. These three test cases were not used to choose the fix or a checkpoint. They are a small diagnostic, not the earlier full nine-case evaluation.

| Policy | Mean final coverage | Mean coverage AUC | Mean team return | Mean discounted return | Same-state action change |
|---|---:|---:|---:|---:|---:|
| Frozen CNN | .9225 | .8166 | -52.320 | -4.354 | 0 |
| Stabilized GCN | .9075 | .8108 | -48.343 | -2.757 | .0259 |
| Stabilized plain LAMP | .9042 | .8045 | -42.236 | -2.476 | .0401 |

Both corrected models had lower mean coverage and higher mean reward than the reference on these three cases. This is an observed objective tradeoff under this small fixed sample, not a causal explanation or a statistically supported general result. Both policies changed actions on encountered states, so they did not merely copy CNN; nevertheless, those changes did not improve mean coverage. Independent-test mean raw hidden RMS was 0.383 for LAMP; actor saturation remained about 31% for LAMP and 43% for GCN. Mean Q spans were about 0.02–0.03, so the critics can numerically distinguish the enumerated actions, but a nonzero span does not establish correct action ranking.

Warm-start holdout one-step SARSA TD MSE was roughly .011–.012 both before and after. Discounted realized-return MSE remained about 55–58 while the mean realized discounted return was -7.13. In particular, this does not demonstrate accurate value estimates, and the 20-update warm-start should not be described as having solved critic calibration.

## Decision and remaining uncertainty

Confirmed: old GCN collapse coincided with a near-constant, very large residual and high blockage; old LAMP accumulation began at initialization; the new controlled configuration kept raw recurrent magnitudes much smaller in these trajectories and passed the predeclared short-run stopping check. Unverified: the unique historical cause of GCN collapse, the separate causal effect of each stabilization, critic action-ranking accuracy, and performance across further seeds. The observed reward/coverage differences cannot establish a general tradeoff.

Do **not** resume lifecycle/ECC ablations as if the baseline were now validated: critic return prediction is still poor, actor saturation remains substantial, and the independent test has only three cases. The next smallest investigation is to evaluate team-critic calibration and counterfactual action ordering on fixed held-out trajectories; do not add a research module or alter rewards in this pilot.

## Commands and integrity

Run from the project root with `bash scripts/run_flight_residual_stability.sh`; `PYTHON_BIN` may select a compatible interpreter. `--residual-mode unbounded`, `--residual-bound` and `--xi` are supported, but changing them requires a fresh output directory/configuration; the existing run refuses to mix mismatched configurations or optimizers. Raw diagnosis, fixed manifests, warm-start data and metrics, initial/periodic/last checkpoints, validation trajectories, per-update KL decisions, independent test and audit are in `outputs/flight_residual_stability/`.

All 142 environment, original LAMP, previous flight and new stability tests passed. The prior flight pilot's 196-file set and hashes were unchanged, the frozen CNN hash remained unchanged, and there were no non-active offloads in the pilot. The source paper's eight baseline hashes and the old dynamic-fleet source's 94 snapshot hashes were also rechecked unchanged. Outputs are deliberately ignored by Git, while code, tests and this report are committed.
