# Dynamic fleet V0 diagnostic report

## Result

On the existing `SAGIN` conda environment, the fixed six-slot environment completed bounded 200-step heuristic and pretrained-actor diagnostics. These are hand-configured state-machine checks, not realistic or comparative experiments. The original CNN and GCN actor checkpoints were loaded and executed without training; the diagnostic uses greedy argmax actions. No abnormal exhaustion was recorded in either run.

The heuristic run started with four search-active aircraft and two reserves; its active count ranged from 1 to 6. It recorded 13 reserve launches, 15 return starts and 14 completed charges. The following sequence is from `data/dynamic_fleet_smoke/events.jsonl`:

| Step | Event |
|---:|---|
| 1 | UAV 4 standby → active, at vehicle (10,10) |
| 3 | UAV 0 active → returning, at (7,12) |
| 5 | UAV 0 reached (10,10) through individual return moves and entered charging |
| 12 | UAV 0 charging → standby, at full 360,000 J |
| 16 | UAV 0 standby → active again |

The pretrained diagnostic recorded 14 launches, 17 return starts, 15 completed charges and active count 1–6. Both used seed 42, initial UAV 0 energy 57%, thresholds 0.8/0.6 and 30,000 J/step charging to induce cycles. These are separate from normal dataclass defaults. Counts and uncertainty are diagnostic outputs, not evidence of superiority or controlled performance comparisons.

## Verification

`baseline_check.json` records the repeated original-environment baseline: Python 3.12.5, torch 2.4.1, PyG 2.6.1, four-aircraft observation `[4,3,15,15]`, eight graph nodes and 24 directed edges; both actor forward passes succeeded. `tests/test_dynamic_fleet.py` covers fixed slots, status masks, no standby/returning search or offload, return movements and exhaustion handling, FIFO and charging, slot mapping and original environment compatibility. Per-step checks in each smoke runner additionally verify finite state/graph values, energy limits, charging monotonicity, search attribution, inactive graph isolation, offload masks, and individual return moves.

The original project's eight requested key files match their initial checksums in `SOURCE_FINAL_SHA256_CHECK.txt`. This check is limited to those files; it is not a checksum of every source file. Source, model and GA files in the new copy are unchanged. The original visualization assumes a `done` flag implies a one-time terminal lifetime; use the new event/snapshot logs for dynamic status rather than interpreting that visualization's existing lifetime counter as a dynamic fleet metric.

## Limitations

The design does not replenish toward a desired fleet size when no new low-energy request is pending: despite available standby UAVs, the heuristic run drops to one search-active UAV for extended periods. It does not optimize charging placement, manage realistic charging hardware, prove safety for all wind or energy settings, account for failures or retrain policies. The exact-cell stepped energy estimator is new logic, using the original wind and flight-energy equations; it is not the original approximate return estimator. No claim is made that the diagnostic configuration or coverage is representative. A minimal next step is to define and test an explicit target-active-fleet replenishment rule, including contention and charging-capacity scenarios, without conflating that new rule with the current V0.
