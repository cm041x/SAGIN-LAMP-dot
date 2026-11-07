# Graph-memory-assisted flight residual pilot

The experiment keeps V1.5 dynamics, rewards, energy, fixed GA deployment,
target-aware fleet rules and one 600 W charging slot. It requests HAPS for
every eligible offload, preserving the environment's feasibility/local-fallback
logic. This isolates flight decisions and does not establish HAPS optimality.

The frozen original CNN supplies its last ReLU hidden vector and eight-action
probabilities. A zero-output residual head consumes that vector plus a graph
embedding and adds logits to `log(clamp(pi_base, 1e-8))`. All variants use
the same current lifecycle context, graph and CNN-local features; infrastructure
local-feature placeholders are zero. Actor and both critics see the CNN local
features. The inherited graph task-difficulty feature is not refreshed; the
known environment semantics remain in `ENVIRONMENT_SEMANTICS_AUDIT.md`.

The decision snapshot is immediately before each physical flight step. Its
mask contains pre-flight active UAVs. The actor's proposed action remains in
replay even when the fleet safety rule overrides it; execution type and reason
are recorded. The environment's post-flight `eligible` set is a separate
offload mask. The team reward is that step's flight reward plus its fixed-HAPS
offload reward, with one discount per physical step. Individual returns are
not team terminals. Memory is advanced once per pre-flight snapshot and raw
episode prefixes are reconstructed separately by actor, critics and targets.

Four trained variants use an eight-action joint critic, sentinel 8 for inactive
slots, team SAC with fixed alpha 0.01, and optionally the inherited eight-class
event-conditioned critic. The fifth policy is the untrained frozen CNN.
The 20-episode/200-step training budget and 500-step validation/evaluation
horizon are deliberately different and do not establish convergence or
generalization. Model parameter updates, action changes on encountered states,
trajectory changes and performance differences are reported independently.
