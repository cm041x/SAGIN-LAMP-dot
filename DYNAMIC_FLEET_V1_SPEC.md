# Target-aware dynamic fleet V1

V1 is a new `DynamicFleetV1Env` subclass; the tagged V0 environment remains unchanged. It retains six stable UAV slots, the original pretrained CNN/GCN, terrain, SAGIN links and physics, and adds a pure, separate `FleetScheduler` with three modes:

- `no_replacement`: four initial searchers; no reserve ever launches or relaunches, including after charging.
- `naive_threshold`: each low-energy active UAV independently requests a standby UAV, once per successful binding; there is no global automatic replenishment. It matches V0 behavior when their charging increments match.
- `target_aware`: all pending active return requests and current active UAVs feed a single deficit calculation. The target defaults to four but is configurable.

```text
active = count(status == ACTIVE)
pending = count(active UAVs with a return request, assigned or not)
projected = active - pending
deficit = max(0, target - projected)
candidates = eligible standby UAVs sorted by (-energy, standby_since_step, ID)
launch up to min(deficit, len(candidates))
assign launches one-to-one to oldest unassigned pending requests first
mark any remaining launch restore_target_count or post_charge_replenishment
return due aircraft after handover, immediately if a safety boundary is reached
```

An aircraft remains active through handover; a 4→5→4 transition is therefore intentional. Already assigned pending requests remain in `pending`, preventing duplicate replacement. Bindings are released after the original aircraft enters return, or immediately if the replacement itself must return first. Decisions are returned as `FleetDecision`; the scheduler never mutates UAV objects. A launch is eligible only at energy ratio at least 0.95 by default. The environment applies decisions and logs request, binding, unavailability, launch reason, status and release events.

Physical charging uses `min(capacity-energy, power_w*efficiency*T)` joules per occupied step. Grid energy is the actual stored increment divided by efficiency, including a partial final step. The default 600 W and 0.90 efficiency are nominal simulation parameters, not verified hardware. V0 retains its historical `charge_rate_per_step`; it is ignored in V1. V0's fast 30 kJ/step diagnostic and explicitly configured charging unit tests are separate from V1 formal runs.

For post-step search counts `n_t` and target `K`, integrated deficit is `sum(max(0,K-n_t))`, surplus is `sum(max(0,n_t-K))`, and target-met is `n_t >= K`. Coverage AUC is the mean per-step coverage; replacement delays include assigned requests only. Unfinished target recovery episodes are right-censored, not assigned zero delay. `unresolved_return_requests` is derived from events as returns without an assigned replacement plus active unassigned requests; the original active-pending counter is retained separately. Service request and effective target counts are distinguished; effective fallback follows the original feasibility equations. UAV energy is broken down into search flight, return flight, local processing and communication, with grid charging input kept separate.

This is rule-based control, not a learned policy. The vehicle is stationary, charge power is idealized, return safety is a one-step check rather than a formal guarantee, search count alone does not ensure coverage, and the existing SAGIN model is retained. The experiment does not prove performance on held-out deployments or charging hardware.
