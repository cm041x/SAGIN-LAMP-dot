# Inherited V1.5 environment semantics

No environment physics or reward has been modified in this prototype.

- Search uncertainty is updated during the flight step, before the offload
  action is processed. The paper's description associates effective sensing
  progress with successful task processing; this inherited ordering is a
  paper/code discrepancy and can limit interpretation of offload effects.
- `generate_sagin_graph()` places a UAV's task difficulty in its node feature
  from its position when the graph is generated. `get_obs_2()` copies `self.G`
  and updates edges, but does not refresh that feature as the UAV moves.
  V1.5 regenerates `self.G` at reset. Thus the inherited graph task feature
  may reflect the reset position while `step_offload()` uses task difficulty
  at the current position. Lifecycle context supplies current positions and
  energy, but does not silently repair the stale task feature.
- Physical graph edges and service fallback retain the V1.5 definitions. The
  model uses centralized lifecycle context without simulating telemetry cost.
- Fleet rotations alter the meaning of one-time UAV lifetime. Final coverage
  after 500 steps in a rechargeable fleet is not directly comparable to a
  fixed-battery original-paper result with different stopping conditions.

These observations are limitations of the inherited environment, not results
of a changed physical model. They should be addressed in a separately scoped
experiment with the old and revised environments compared explicitly.
