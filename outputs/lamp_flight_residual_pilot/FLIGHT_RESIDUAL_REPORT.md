# Graph-assisted flight residual pilot

Short pilot only: 20 × 200 training steps versus 500-step evaluation; one training seed.

| Policy | Coverage | Coverage AUC | Repeat | UAV energy J | Return energy J | Deficit | Changed actions | Changed trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_cnn_reference | 0.9236 | 0.8187 | 0.7803 | 9205269 | 1582977 | 338.7 | 0.0000 | 0/9 |
| gcn_flight_residual | 0.3500 | 0.3339 | 0.9204 | 9154765 | 1719396 | 284.8 | 0.9374 | 9/9 |
| lamp_plain_flight_residual | 0.8972 | 0.7960 | 0.7923 | 9278227 | 1488926 | 288.0 | 0.0519 | 9/9 |
| lamp_lifecycle_flight_residual | 0.8986 | 0.7988 | 0.7754 | 9342062 | 1592449 | 393.6 | 0.0998 | 9/9 |
| lamp_lifecycle_ecc_flight_residual | 0.9069 | 0.8022 | 0.7750 | 9286864 | 1654917 | 389.8 | 0.1080 | 9/9 |

## Training diagnostics

- gcn_flight_residual: {'episode': 20, 'environment_steps': 4000, 'updates': 360, 'parameter_changes': {'actor': True, 'q1': True, 'q2': True}, 'actor_encoder_changed': True, 'residual_weight_delta_norm': 1.5493085384368896, 'mean_actor_encoder_gradient': 7.374551445881666e-06, 'mean_residual_gradient': 0.0016015703801249318, 'mean_critic_encoder_gradients': [0.05145939160298035, 0.05220754865412244], 'mean_actor_loss': 0.08709608231308974, 'mean_critic_loss': 0.005911622261919547, 'mean_conditioner_gradient': None, 'event_counts_both_critics': [0, 0, 0, 0, 0, 0, 0, 0], 'event_ce': None, 'event_brier': None, 'event_accuracy': None, 'majority_accuracy': None}
- lamp_plain_flight_residual: {'episode': 20, 'environment_steps': 4000, 'updates': 360, 'parameter_changes': {'actor': True, 'q1': True, 'q2': True}, 'actor_encoder_changed': True, 'residual_weight_delta_norm': 0.7264108061790466, 'mean_actor_encoder_gradient': 0.00010004654151436586, 'mean_residual_gradient': 0.0014816152321924453, 'mean_critic_encoder_gradients': [0.08688791406448218, 0.09062133508027223], 'mean_actor_loss': 0.1668543047996031, 'mean_critic_loss': 0.009391706094579098, 'mean_conditioner_gradient': None, 'event_counts_both_critics': [0, 0, 0, 0, 0, 0, 0, 0], 'event_ce': None, 'event_brier': None, 'event_accuracy': None, 'majority_accuracy': None}
- lamp_lifecycle_flight_residual: {'episode': 20, 'environment_steps': 4000, 'updates': 360, 'parameter_changes': {'actor': True, 'q1': True, 'q2': True}, 'actor_encoder_changed': True, 'residual_weight_delta_norm': 0.6583551168441772, 'mean_actor_encoder_gradient': 6.469081549340268e-05, 'mean_residual_gradient': 0.0011860544510473327, 'mean_critic_encoder_gradients': [0.09350070407930622, 0.09906884719807837], 'mean_actor_loss': 0.1581115992170655, 'mean_critic_loss': 0.009401111551657475, 'mean_conditioner_gradient': 1.217300721123111e-07, 'event_counts_both_critics': [0, 0, 0, 0, 0, 0, 0, 0], 'event_ce': None, 'event_brier': None, 'event_accuracy': None, 'majority_accuracy': None}
- lamp_lifecycle_ecc_flight_residual: {'episode': 20, 'environment_steps': 4000, 'updates': 360, 'parameter_changes': {'actor': True, 'q1': True, 'q2': True}, 'actor_encoder_changed': True, 'residual_weight_delta_norm': 0.6271395683288574, 'mean_actor_encoder_gradient': 9.147393594556455e-05, 'mean_residual_gradient': 0.0016774788095455246, 'mean_critic_encoder_gradients': [0.1327237248969346, 0.15273293321483258], 'mean_actor_loss': 0.07616732252645306, 'mean_critic_loss': 0.08507895415855779, 'mean_conditioner_gradient': 2.081430374640887e-07, 'event_counts_both_critics': [37630, 2278, 2300, 50, 1806, 102, 0, 0], 'event_ce': 0.6863902939006707, 'event_brier': 0.29779378524987976, 'event_accuracy': 0.8380201966608261, 'majority_accuracy': 0.8520128605714803}

## Evaluation action, safety and event details

- frozen_cnn_reference: 0/15396 proposals differ from the frozen CNN on encountered states; 9/15396 overridden by safety boundary; requested 0..4=[0, 0, 15145, 0, 0], effective 0..4=[582, 0, 14563, 0, 0].
- gcn_flight_residual: 14717/15701 proposals differ from the frozen CNN on encountered states; 27/15701 overridden by safety boundary; requested 0..4=[0, 0, 15453, 0, 0], effective 0..4=[289, 0, 15164, 0, 0].
- lamp_plain_flight_residual: 802/15707 proposals differ from the frozen CNN on encountered states; 13/15707 overridden by safety boundary; requested 0..4=[0, 0, 15448, 0, 0], effective 0..4=[366, 0, 15082, 0, 0].
- lamp_lifecycle_flight_residual: 1378/14828 proposals differ from the frozen CNN on encountered states; 32/14828 overridden by safety boundary; requested 0..4=[0, 0, 14571, 0, 0], effective 0..4=[425, 0, 14146, 0, 0].
- lamp_lifecycle_ecc_flight_residual: 1528/14849 proposals differ from the frozen CNN on encountered states; 22/14849 overridden by safety boundary; requested 0..4=[0, 0, 14597, 0, 0], effective 0..4=[405, 0, 14192, 0, 0].
  ECC independent evaluation: mean CE=0.62640, Brier=0.27825, accuracy=0.84622, per-run majority reference=0.84622; both critics counted.

CNN weights are frozen and hashed; action masks use pre-flight active slots. Flight proposals overridden by a return rule remain in replay.
The fixed offload request is HAPS, with the original environment fallback. Requested and effective counts are in each run.
Parameter change does not establish action change; action change does not establish performance gains.
The inherited stale graph task feature and uncertainty-before-offload behavior remain unmodified.
Previous offloading-only four variants chose HAPS in the evaluated states; this is not proof of global optimality.
ECC scores are sampled training diagnostics, not independent calibration; no novelty or convergence claim.
