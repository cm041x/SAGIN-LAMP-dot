# Dynamic HCSAC baseline report — IN PROGRESS

This is not a validated formal baseline. The scratch and finetune pilot runs, exact episode-boundary resume check, 100-step three-policy pilot, and existing-environment regression checks have completed. All requested formal training and the 210-run formal evaluation must finish before any scientific comparison or acceptance claim can be made.

## Verified pilot

- Scratch seed 3001 completed 10 × 50 steps and resumed from episode 10 to 15, restoring step 500 and replay sizes 1993/500. The 15-episode resumed run matched a separate uninterrupted run in every recorded history field except elapsed time and in final CNN/GCN actor and critic hashes.
- Finetune seed 3001 loaded both original actors/critics/target critics and log temperatures with fresh optimizers; exact CNN and GCN forward equivalence was checked. Both actor and both critic fingerprints changed during its 10 × 50-step pilot.
- Zero-shot, scratch-pilot best and finetune-pilot best completed a 100-step execution check under wind 4800/eval 42 with no inactive UAV offload; zero-shot step-100 coverage matched V1.5. These pilot values are not a formal test result and were not used to select a checkpoint or adjust hyperparameters.
- The existing V0 smoke, V1 short run and both V1.5 pilots ran again in separate regression output directories; the original baseline loaded both actors. New CNN replay records include active-before/executed-slot witnesses for audit.
- A 10-update batch-128 CPU microbenchmark measured ~0.206 s for both SAC branches combined, corresponding to ~171 hours for requested formal updates alone. Actual step and validation/checkpoint time will add to that estimate. Formal progress must be read from checkpoints/history, not inferred from this estimate.

## Outstanding

Six 1000-episode formal training runs, their 20 validations each, 210 formal evaluation rollouts, complete cross-run statistical analysis, formal independent audit and acceptance report remain outstanding until verified from actual artifacts. No claim is made about convergence, generalization, comparative performance or the necessity of a new architecture.
