# Control and information audit

The audited implementation is the fixed-slot V1.5 environment at source commit `d17382c123eb32fa811e78e9e159936d766e9bc3`. It is not a claim that the paper's GCN cannot handle changing topologies.

| Signal | Existing implementation | This rule experiment |
|---|---|---|
| Existence/padding | Six real UAV nodes plus four infrastructure nodes; all existence flags true. | Same six real slots, no fictitious node or padding. |
| Flight eligibility | Pre-flight `status == active`; fleet safety can override a proposal. | Same for all P0/P1/P2; inactive aircraft do not receive policy flight actions. |
| Offload eligibility | Post-flight `env.eligible`; new launches and returns are not necessarily eligible that slot. | Identical HAPS request and original local fallback. |
| Physical graph | Inactive UAV node features are zero and all their physical offload edges are removed. | Unchanged; no charger or telemetry edges are introduced. |
| Lifecycle data | A separate context contains per-slot status, position, energy and age. Queue length and charging utilization are broadcast to all node contexts. | A centralized current telemetry array is available to every controller; P0's fixed CNN interface continues to read only its local input. |
| Search information | Active flight CNN receives 15x15 uncertainty/occupancy and wind observations. | A centralized controller may fuse only current active local views. Unknown/occluded cells are NaN, not privileged ground truth; all policies receive the same observation object. |

This is a centralized information assumption; it does **not** simulate telemetry links, telemetry cost, a decentralized protocol, or future visibility. P0 is a deliberately frozen local policy and cannot use all data exposed to the planning controller; P1-P0 therefore measures the combined effect of coordinated processing of the commonly exposed information and the different policy class. It is not a controlled architecture-only comparison. P1 and P2 receive exactly the same current information, candidates, energy feasibility, matching and movement budget. A truly local-only comparison or a full-map oracle is not silently folded into these results.

## Controlled forward sensitivity

`scripts/audit_availability_information.py` performs deterministic, untrained-model forward checks at a real environment snapshot. With all edges and active state fixed, changing a charging UAV's energy from .2 to .8 of battery capacity changes its lifecycle context and explicit telemetry, but changes the active-node embedding of both the existing GCN and LAMP encoders by **0.0**. Changing only the age of a waiting UAV from 2 to 8 steps also produces **0.0** active-node embedding change. The explicit current-telemetry feature changes by .6 and .012 respectively; the approximate ready score changes from 1.467 to 1.867 in the energy test. Full values are in `results/availability_gap/sensitivity.json`.

These tests show the specific per-aircraft information path is absent under the physical graph at this snapshot. They do **not** imply that all information about the ground fleet is absent: changing charger occupancy or queue length alters aggregate fields broadcast to active nodes. The tests examine untrained encoder representations, not a learned action response or performance. The old representation remains unchanged; the new availability-aware rule uses an explicit centralized telemetry interface, not extra physical communication edges.

## Planner and forecasting boundary

Both planners match the same currently visible uncertain candidate cells (at most 96) with the same rectangular deterministic matching algorithm, same 8-neighbor next-step solver, same observed current wind/energy feasibility check, and one matching call per step. P1 scores current distance only. P2 adds a fixed 30-step, model-based estimate of active persistence and current-ground-fleet recovery to adjust candidate costs. It assumes current wind cost, one-slot FIFO and constant charging power, without knowing future flight actions, future arrivals, exact return-energy exhaustion or future events. Its computed time to readiness is an estimate, not a ground-truth arrival time. `handover_mode=none`: the existing V1.5 same-region waypoint code is not used or described as future-supply planning.

Current-view fusion can include cells visible to another active UAV but not to a given local CNN input; only a shared centralized controller under the stated assumption is modeled. The frozen CNN uses that shared observation object only through its existing local channel. This is an information-processing capability limitation of P0 and prevents attributing all P1-P0 gains to one isolated coordination mechanism.
