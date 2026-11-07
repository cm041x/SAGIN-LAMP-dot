# LAMP prototype: scope and experiment protocol

This is an exploratory offloading-branch prototype. V1.5 physics, rewards,
target-aware rule scheduling, fixed GA deployment and the original pretrained
CNN are held fixed. The CNN is frozen and uses argmax. There is no waypoint,
learned fleet scheduler, new flight action or change to communication edges.

Four variants share the same lifecycle context, physical graph, actual
`env.eligible` action mask, team-interval reward, actor update and fixed alpha:
GCN with context, pinned-upstream LAMP propagation, lifecycle-conditioned LAMP,
and lifecycle LAMP with event-conditioned twin critics. The GCN comparison is
not the original per-UAV HCSAC training algorithm.

Independent deterministic initialization streams for the actor, each critic,
and their target copies preserve identical shared LAMP encoder parameters
across the three LAMP variants. The lifecycle conditioner's final layer is
zero at initialization; its first-layer RNG use does not perturb shared heads.

Each node has a stable ID and existence mask. Lifecycle context contains the
documented five-state one-hot, UAV energy and position, elapsed transition and
charging-wait times, charging occupancy, queue length and remaining time.
Infrastructure retains explicit type and appropriate shared current-time
features; inapplicable fields are zero. Context availability is an explicit
centralized telemetry assumption, without communication cost modeling.

The LAMP adapter preserves upstream input injection, skew mixing, separately
normalized symmetric/antisymmetric weighted edges, Euler update and doubled
diagonal in the symmetric parametrization. Zero-initialized lifecycle output
adds only a per-UAV current-context damping adjustment, constant for the inner
Euler iterations. The standalone derivation is in
`docs/LIFECYCLE_LAMP_ECC_DERIVATION.md`. Finite mathematical checks are not
convergence, stability or novelty proofs; in particular xi=1 has no claimed
geometric bound.

The collector holds current offload outcomes pending until the next actual
flight and offload decision. Each team reward is current offload plus next
flight, with an initial prefix and terminal flush. Transition labels use only
authoritative transitions; labels never enter the encoder or actor inputs.
Individual UAV return is not a team terminal. No-action steps remain critic
transitions with entropy zero and actor loss skipped.

Replay stores full raw episodes; unpadded sequences carry explicit all-valid
sequence masks, and padded suffixes are excluded from recurrent processing and
loss. Networks reconstruct their own prefix memory
from the beginning, detach, then backpropagate through contiguous unrolls.
Target networks maintain independent parameters and prefix memory. Pilot is 20
episodes per variant, 200 steps, 2 warmup episodes, 20 updates per subsequent
episode, unroll 16, batch 4, 2 CPU threads; no hyperparameter search. Final
checkpoint validation uses the same nine held-out validation scenarios for each
variant. The separate final evaluation is three fixed wind seeds times three
evaluation seeds. Neither is used to select an intermediate checkpoint.

The V1.5 code marks uncertainty during flight before checking offload
feasibility; the paper describes failed tasks differently. This scope preserves
that behavior and does not claim exact paper reproduction. The same-cell
simultaneous conflict and dropped-task metrics remain unavailable.
