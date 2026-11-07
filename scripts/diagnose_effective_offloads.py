"""Audit requested and effective offloading in the original V1.5 environment."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from Our_experiment.HCSAC.lamp_method.trainer import TeamSAC
from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor
from Our_experiment.HCSAC.lamp_method.policy import SnapshotMemory
from Our_experiment.HCSAC.lamp_method.graph_observation import observe
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import new_env

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/lamp_prototype'
torch.set_num_threads(2);flight=frozen_flight_actor();agent=TeamSAC('gcn_context')
ck=torch.load(OUT/'training/gcn_context/last_checkpoint.pt',map_location='cpu',weights_only=False)
agent.load_state_dict(ck['agent'])
prior=json.loads((OUT/'action_diagnostic.json').read_text())
if not prior['same_joint_action_sequence_per_paired_scenario']:
    raise RuntimeError('Cannot extrapolate one-variant effective counts: action sequences differ')
reference={(r['wind_seed'],r['eval_seed']):r for r in prior['rows'] if r['variant']=='gcn_context'}
rows=[]
for wind in (11,23,4800):
    for seed in (42,43,44):
        env=new_env(500,{'wind_seed':wind,'terrain_seed':seed,'infra_seed':seed})
        state=env._get_obs();memory=SnapshotMemory();digest=hashlib.sha256()
        for _ in range(500):
            ids=np.flatnonzero(env.get_search_active_mask()).tolist();fly_actions=[0]*6
            if ids:
                with torch.no_grad():chosen=flight(torch.as_tensor(state[ids],dtype=torch.float32)).argmax(-1).tolist()
                for i,a in zip(ids,chosen):fly_actions[i]=a
            state,_,done=env.step(fly_actions);snapshot=observe(env)
            with torch.no_grad():
                probs=agent.actor.probabilities(memory.advance(agent.actor.encoder,snapshot))
                actions=[int(probs[i].argmax()) if snapshot.action_mask[i] else 0 for i in range(6)]
            digest.update(bytes(actions[i]+1 if snapshot.action_mask[i] else 0 for i in range(6)))
            _,_,off_done=env.step_offload(actions)
            if done or off_done:break
        if digest.hexdigest()!=reference[(wind,seed)]['joint_action_sha256']:
            raise RuntimeError('Effective-target diagnostic action sequence mismatch')
        requested=[sum(t['requested'][i] for t in env.offload_timeline) for i in range(5)]
        effective=[sum(t['effective'][i] for t in env.offload_timeline) for i in range(5)]
        row={'wind_seed':wind,'eval_seed':seed,'requested':requested,'effective':effective,
             'local_fallback_count':sum(t['local_fallback_count'] for t in env.offload_timeline)}
        rows.append(row);print('effective audit',wind,seed,row,flush=True)
total_req=[sum(r['requested'][i] for r in rows) for i in range(5)]
total_eff=[sum(r['effective'][i] for r in rows) for i in range(5)]
data={'representative_variant':'gcn_context','scenarios':len(rows),'same_joint_actions_all_variants':True,
      'requested_totals':total_req,'effective_totals':total_eff,
      'local_fallback_total':sum(r['local_fallback_count'] for r in rows),'rows':rows}
(OUT/'effective_offload_diagnostic.json').write_text(json.dumps(data,indent=2)+'\n')
with (OUT/'PROTOTYPE_REPORT.md').open('a') as f:
    f.write('\nRepresentative effective-target audit for the nine paired scenarios: '
            f'requests 0..4 = {total_req}, effective targets 0..4 = {total_eff}, '
            f"local fallbacks = {data['local_fallback_total']}. "
            'The audit reproduced the exact GCN action hashes; all four variants have the same paired action sequence.\n')
