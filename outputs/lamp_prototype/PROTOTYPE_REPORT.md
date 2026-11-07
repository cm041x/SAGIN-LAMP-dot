# LAMP prototype report

Short pilots only; no claim of superiority, novelty, stability or global path coordination.
All 36 evaluation rollouts use fixed pretrained CNN actions and final pilot checkpoints. 

| Variant | Coverage | Coverage AUC | Repeat ratio | UAV energy J | Deficit |
|---|---:|---:|---:|---:|---:|
| gcn_context | 0.9236 | 0.8187 | 0.7803 | 9205269 | 338.7 |
| lamp_plain | 0.9236 | 0.8187 | 0.7803 | 9205269 | 338.7 |
| lamp_lifecycle | 0.9236 | 0.8187 | 0.7803 | 9205269 | 338.7 |
| lamp_lifecycle_ecc | 0.9236 | 0.8187 | 0.7803 | 9205269 | 338.7 |

## Training and event diagnostics

Event metrics are sampled training-batch diagnostics, not calibration on independent data. Counts include both independent critics.
- gcn_context: episodes=20, env_steps=4000, updates=360, seconds=254.2, peak_rss_bytes=374538240, actor_tensors_changed=8, event_counts=[0, 0, 0, 0, 0, 0, 0, 0], event_CE=None, event_Brier=None, batch_accuracy=None, majority_class_reference=None.
- lamp_plain: episodes=20, env_steps=4000, updates=360, seconds=1042.8, peak_rss_bytes=514441216, actor_tensors_changed=10, event_counts=[0, 0, 0, 0, 0, 0, 0, 0], event_CE=None, event_Brier=None, batch_accuracy=None, majority_class_reference=None.
- lamp_lifecycle: episodes=20, env_steps=4000, updates=360, seconds=1751.7, peak_rss_bytes=555745280, actor_tensors_changed=14, event_counts=[0, 0, 0, 0, 0, 0, 0, 0], event_CE=None, event_Brier=None, batch_accuracy=None, majority_class_reference=None.
- lamp_lifecycle_ecc: episodes=20, env_steps=4000, updates=360, seconds=2352.8, peak_rss_bytes=555745280, actor_tensors_changed=14, event_counts=[37568, 2360, 2310, 38, 1812, 78, 0, 0], event_CE=0.6770700160000059, event_Brier=0.2963858297715584, batch_accuracy=0.8463450645407041, majority_class_reference=0.8506090657972196.

## Boundaries

The dense parity check independently reconstructs the pinned upstream operations; it does not execute official torch_scatter.
The default xi=1 has no claimed geometric stability guarantee. Validation-selected or formal learning outcomes are not established.
The V1.5 environment updates uncertainty in flight before offload feasibility; this differs from the paper description and was not changed.
The stopped Dynamic HCSAC run has a known temperature configuration concern and is not repaired by this work.

## Saved-checkpoint audit

Pilot status: passed; details in `audit.json`.

Classifier event counts above apply to ECC training batches; regular critics have no event head. The following counts are actual collected transition labels, including ordinary-critic variants. Weighted event scores use the same sampled-event counts as the majority reference.

- gcn_context: mean actor loss 0.04568492094889128, critic loss 0.009193385098802133; changed actor/critic1/critic2 tensors 8/11/11; conditioner gradient None; weighted event CE None; weighted event accuracy None; weighted event Brier None; majority one-hot Brier reference None; collected event counts [3460, 187, 189, 3, 153, 8, 0, 0].
- lamp_plain: mean actor loss 0.14309151447895502, critic loss 0.010088002844066876; changed actor/critic1/critic2 tensors 10/13/13; conditioner gradient None; weighted event CE None; weighted event accuracy None; weighted event Brier None; majority one-hot Brier reference None; collected event counts [3451, 192, 194, 0, 156, 7, 0, 0].
- lamp_lifecycle: mean actor loss 0.14379504404755103, critic loss 0.010092788541482555; changed actor/critic1/critic2 tensors 14/17/17; conditioner gradient 1.917057418746963e-05; weighted event CE None; weighted event accuracy None; weighted event Brier None; majority one-hot Brier reference None; collected event counts [3455, 188, 195, 0, 151, 11, 0, 0].
- lamp_lifecycle_ecc: mean actor loss 0.14851118414404077, critic loss 0.08964545929597484; changed actor/critic1/critic2 tensors 14/17/17; conditioner gradient 1.9458600135721956e-05; weighted event CE 0.6769300676156541; weighted event accuracy 0.8464203239653263; weighted event Brier 0.29625696075009583; majority one-hot Brier reference 0.29878186840556076; collected event counts [3447, 194, 193, 3, 153, 10, 0, 0].

Conditioner final-layer weight delta norms from deterministic initialization:

- lamp_lifecycle: {'actor': 0.6039861440658569, 'q1': 0.3498925268650055, 'q2': 0.3254750072956085}.
- lamp_lifecycle_ecc: {'actor': 0.4939403235912323, 'q1': 0.4568586051464081, 'q2': 0.3745232820510864}.

## Coincident evaluation diagnostic

Independently replayed 36 evaluations. Identical joint-action sequences within each paired scenario: True.

- gcn_context requested offload action counts 0..4: [0, 0, 15145, 0, 0].
- lamp_plain requested offload action counts 0..4: [0, 0, 15145, 0, 0].
- lamp_lifecycle requested offload action counts 0..4: [0, 0, 15145, 0, 0].
- lamp_lifecycle_ecc requested offload action counts 0..4: [0, 0, 15145, 0, 0].

Action 2 requests HAPS; requested actions can be converted to local processing by the inherited environment. Matching actions under the same frozen CNN and environment explain matching physical outcomes, not equivalent learned parameters.

Representative effective-target audit for the nine paired scenarios: requests 0..4 = [0, 0, 15145, 0, 0], effective targets 0..4 = [582, 0, 14563, 0, 0], local fallbacks = 582. The audit reproduced the exact GCN action hashes; all four variants have the same paired action sequence.
