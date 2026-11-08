"""Validate the corrected 500-step integration and real gradient path."""
import copy
import json
import torch
from Our_experiment.HCSAC.flight_stability.policy import VARIANTS
from Our_experiment.HCSAC.flight_stability.trainer import TeamSAC
from Our_experiment.HCSAC.flight_stability.collector import collect_episode
from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor
from Our_experiment.HCSAC.flight_stability.diagnose import OUT,jwrite
from Our_experiment.HCSAC.training.checkpointing import module_hash

torch.set_num_threads(2)
cnn=frozen_flight_actor();before=module_hash(cnn);case={'wind_seed':1180,'terrain_seed':900000,'infra_seed':900000}
ref=collect_episode(None,cnn,case,500,greedy=True);tr=ref.transitions
probe={'snapshots':[x['snapshot'] for x in tr[:16]],'indices':[0,5,10,15]};results={}
for variant in VARIANTS:
    torch.manual_seed(3001);agent=TeamSAC(variant)
    observed=collect_episode(agent.actor,cnn,case,500,greedy=True)
    same=[x['positions'] for x in observed.trajectory]==[x['positions'] for x in ref.trajectory]
    initial=copy.deepcopy(agent.actor.state_dict())
    critic=agent.update([(tr[:5],tr[5:9])],actor_enabled=False)
    actor=agent.update([(tr[:5],tr[5:9])],probe=probe)
    changed=any(not torch.equal(v,agent.actor.state_dict()[k]) for k,v in initial.items())
    results[variant]={'initial_trajectory_equal':same,'actor_changed':changed,'guard':actor['actor_guard'],
                      'critic_loss':critic['critic_loss'],'actor_loss':actor['actor_loss'],
                      'hidden_rms':observed.metrics['mean_raw_hidden_rms']}
    assert same and changed and actor['actor_guard']['accepted'] and actor['actor_guard']['kl']<=.01
assert module_hash(cnn)==before
jwrite(OUT/'preflight.json',{'passed':True,'cnn_unchanged':True,'models':results})
print(json.dumps(results,indent=2))
