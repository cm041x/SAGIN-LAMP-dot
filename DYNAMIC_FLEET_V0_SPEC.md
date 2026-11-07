# Charging-vehicle-assisted dynamic fleet V0

## Scope and assumptions

The fixed-fleet original runs per-aircraft CNN search and GCN SAGIN offloading. Its original energy exhaustion branches place an aircraft at its destination immediately. V0 uses the original grid, sampled wind, terrain, flight power, infrastructure and copied offload equations, but substitutes a physically stepped return and charging state machine. The charging vehicle is a single fixed grid point, independent of the existing GBS/HAPS/LEO/Cloud. No GA, weight, network or original environment file is modified. This is an environment integration diagnostic, not a trained algorithm or performance claim.

## Slots, state and ordering

The default six fixed slots have four active and two standby UAVs. `Status` defines standby, active, returning, waiting_charge and charging. Non-active observations are zero; occupancy in the original observation excludes non-active slots. At reset, standby positions are the vehicle and do not mark its cell searched. The aircraft initially in flight use the original initial positions unless explicit positions are passed. `active_mask` means search active; airborne means active or returning.

Each step first checks energy and launches at most one standby for each unserved low-energy request. Newly launched UAVs start at the vehicle and become eligible for search/offload on the next step. Searchers below the return threshold return after handover, unless the wind-aware safety boundary forces earlier return. The boundary includes the proposed flight, a conservative local-compute estimate and a margin on a gridwise return path. Return moves one of eight grid directions toward the vehicle, using original wind and `_flight_energy`; they neither search nor offload. Insufficient energy records an abnormal event without negative energy or teleporting. On arrival a FIFO queue assigns available charging slots. A charging aircraft gains energy on subsequent steps, capped at the original initial energy, then becomes standby and clears its replacement flags. It may launch again.

`step` is followed by `get_obs_2` and `step_offload`, matching the original online structure. The adapted service equations consider only aircraft eligible for that step; other rewards are zero. The graph always contains six UAV nodes followed by four original infrastructure nodes with six features; inactive nodes have zero features and no edges, with explicit `slot_id`. Original graph feature semantics are retained: UAV difficulty features are based on the graph generated at reset; dynamic GBS/HAPS links are refreshed by the original method.

Termination occurs on the horizon, remaining time or an optional uncertainty threshold, not on temporary lack of active UAVs. No automatic deadlock termination is implemented. Event and per-step snapshot output includes counts, energy, positions and statuses; `return_move` contains its origin.

## Defaults, diagnostic overrides and limitations

The dataclass contains normal defaults. The smoke runner separately uses central initial positions, lower UAV 0 energy (57%), thresholds 0.8/0.6 and 30 kJ per step to expose cycles. These settings are diagnostics, not calibrated field conditions. No vehicle motion, heterogeneity, failures, network disruptions, task allocation, route optimization or policy training is modeled. The threshold policy does not guarantee a minimum search fleet: the observed smoke run later has one active UAV despite charged standby slots. Reserve scheduling, conservative safety under arbitrary parameters, and field-scale evaluation remain open. The energy boundary is a one-step safeguard, not a proof of safety against every overload or pathological configuration; exhaustion is surfaced explicitly.
