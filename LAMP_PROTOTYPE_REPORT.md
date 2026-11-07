# LAMP offloading prototype report

## Scope and status

Four isolated 20-episode pilots and 36 fixed 500-step evaluation rollouts
completed on CPU with two threads. Every pilot collected 4,000 environment
steps and performed 360 updates after two warmup episodes. The original CNN
actor was frozen; V1.5 physical dynamics, reward, target-aware scheduler, GA
deployment and charging assumptions were preserved. These are short diagnostics,
not a paper-scale experiment, new fleet scheduler or global path coordination.

The official LAMP source is pinned to
`69f6eb53316ef19296fce4fd25e0a57cb128c231` and retains its MIT license.
Since `torch_scatter` was unavailable, the PyTorch adapter was compared against
an independent dense reconstruction, not a direct official execution. The
maximum checked float32 error was 1.1921e-7 (tolerance 2e-6).
The complete staged diff has three blank-line-at-EOF warnings in upstream YAML
files. Those official files are intentionally preserved byte-for-byte; the
staged diff excluding `external/LAMP/` passes `git diff --check`.

The total launcher passed 97 inherited regression tests and 28 prototype tests,
four 500-step smoke rollouts, the analytical numerical checks and checkpoint
restoration. The maximum random float64 advantage identity error was 2.22e-16;
the action-independent event-distribution term was zero and the positive-target
temperature direction checks passed. These establish only the checked algebra
and implementation properties, not convergence or arbitrary-step stability.

## Fixed evaluation: 9 paired scenarios per method

Mean ± sample standard deviation over wind seeds 11, 23, 4800 and evaluation
seeds 42–44. A single training seed was used for each method. Coverage is the
inherited 1 − mean uncertainty metric, not an original fixed-fleet result.

| Variant | Final coverage | Coverage AUC | Repeat ratio | Total UAV energy | Fleet deficit |
|---|---:|---:|---:|---:|---:|
| gcn_context | 92.36% ± 2.18 pp | 0.8187 | 0.7803 | 9.205 MJ | 338.7 |
| lamp_plain | 92.36% ± 2.18 pp | 0.8187 | 0.7803 | 9.205 MJ | 338.7 |
| lamp_lifecycle | 92.36% ± 2.18 pp | 0.8187 | 0.7803 | 9.205 MJ | 338.7 |
| lamp_lifecycle_ecc | 92.36% ± 2.18 pp | 0.8187 | 0.7803 | 9.205 MJ | 338.7 |

The equality is not explained by unchanged weights: a full replay of all 36
evaluations found distinct actor parameter hashes but exactly the same masked
joint-action sequence within each matched scenario. Every eligible action was
request 2 (HAPS): 15,145 requests per variant over nine scenarios. A separate
representative re-evaluation reproduced those action hashes and found 14,563
effective HAPS executions and 582 local fallbacks. Thus the deterministic
evaluation did not discriminate among the four trained offload branches; this
does not prove their policy distributions or representations are identical.

All actor and twin-critic parameter tensors changed. For lifecycle and ECC,
the final conditioner-layer weight delta norms were respectively
actor/q1/q2 = 0.6040/0.3499/0.3255 and 0.4939/0.4569/0.3745. Across ECC
training minibatches, event CE was 0.6769, Brier 0.2963 and weighted accuracy
84.64%, versus 85.06% for the majority-label accuracy reference. The two
critics' sampled labels are counted; these are in-training diagnostics, not an
independent calibration assessment. There is no evidence here that event
conditioning improves control, or that the algebraic decomposition is novel.

The four episode-time sums were about 254 s, 1,043 s, 1,752 s and 2,353 s,
respectively. Peak process RSS is a cumulative high-water mark, not an
independent per-method measurement; machine swapping affected elapsed time.
All 36 evaluation runs had zero inactive offloads and zero recorded energy
exhaustion events. See the saved CSV, JSON, action audit and plots in
`outputs/lamp_prototype/` for the full data and provenance.

## Environment and evidence limitations

The inherited graph's UAV task-difficulty node feature may remain tied to the
reset position while the offloading computation uses the current position.
Search uncertainty is also updated before offloading is processed, unlike the
paper's description of failed task progress. Both behaviors were deliberately
left unchanged and verified/documented in `ENVIRONMENT_SEMANTICS_AUDIT.md`.
The centralized lifecycle context assumes telemetry is available without a
separate communication-cost model. Default xi=1 does not satisfy the stated
sufficient bound in the derivation.

The original paper project's eight baseline files match their recorded hashes;
94 normal source/fixed-input files in the previous dynamic project match the
isolation snapshot. The explicitly requested old HCSAC shutdown used SIGINT;
its last loadable checkpoint contains episode 685 and 342,500 environment
steps. Some partial next-episode work may not have been saved. That older run
has an identified temperature configuration concern and is not a formal
performance baseline. No old training process was restarted.

The low-cost next diagnostic is to study why greedy offloading selects HAPS in
every fixed evaluation state, including requested/effective-target and reward
differences, before interpreting architectural performance or enlarging the
training budget.
