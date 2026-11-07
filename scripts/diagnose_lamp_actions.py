"""Re-evaluate and audit action sequences to explain coincident aggregate results."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import torch
from Our_experiment.HCSAC.lamp_method.trainer import TeamSAC
from Our_experiment.HCSAC.lamp_method.collector import collect_episode,frozen_flight_actor
from Our_experiment.HCSAC.training.checkpointing import module_hash

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/lamp_prototype'
VARIANTS=('gcn_context','lamp_plain','lamp_lifecycle','lamp_lifecycle_ecc')
torch.set_num_threads(2);flight=frozen_flight_actor()
prior={(r['variant'],int(r['wind_seed']),int(r['eval_seed'])):r for r in csv.DictReader((OUT/'per_run_metrics.csv').open())}
rows=[]
for variant in VARIANTS:
    ck=torch.load(OUT/'training'/variant/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    agent=TeamSAC(variant);agent.load_state_dict(ck['agent']);actor_hash=module_hash(agent.actor)
    for wind in (11,23,4800):
        for seed in (42,43,44):
            ep=collect_episode(agent.actor,flight,{'wind_seed':wind,'terrain_seed':seed,'infra_seed':seed},500,greedy=True)
            counts=Counter();digest=hashlib.sha256()
            for t in ep.transitions:
                mask=t['snapshot'].action_mask.tolist(); actions=t['actions'].tolist()
                canonical=[actions[i] if mask[i] else -1 for i in range(6)]
                digest.update(bytes(x+1 for x in canonical))
                counts.update(actions[i] for i in range(6) if mask[i])
            old=prior[(variant,wind,seed)]
            if abs(float(old['final_coverage'])-ep.metrics['final_coverage'])>1e-12:
                raise RuntimeError('Replayed coverage differs from saved evaluation')
            rows.append({'variant':variant,'wind_seed':wind,'eval_seed':seed,'actor_sha256':actor_hash,
                         'joint_action_sha256':digest.hexdigest(),'requested_action_counts':[counts[i] for i in range(5)],
                         'final_coverage':ep.metrics['final_coverage']})
            print('action audit',variant,wind,seed,counts,flush=True)
identical=all(len({r['joint_action_sha256'] for r in rows if r['wind_seed']==w and r['eval_seed']==s})==1
              for w in (11,23,4800) for s in (42,43,44))
totals={v:[sum(r['requested_action_counts'][i] for r in rows if r['variant']==v) for i in range(5)] for v in VARIANTS}
result={'replayed':len(rows),'same_joint_action_sequence_per_paired_scenario':identical,
        'requested_action_totals':totals,'rows':rows,
        'note':'action 2 requests HAPS; requested actions are not necessarily effective targets'}
(OUT/'action_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
with (OUT/'PROTOTYPE_REPORT.md').open('a') as f:
    f.write('\n## Coincident evaluation diagnostic\n\n')
    f.write(f"Independently replayed {len(rows)} evaluations. Identical joint-action sequences within each paired scenario: {identical}.\n\n")
    for variant in VARIANTS:f.write(f"- {variant} requested offload action counts 0..4: {totals[variant]}.\n")
    f.write('\nAction 2 requests HAPS; requested actions can be converted to local processing by the inherited environment. '
            'Matching actions under the same frozen CNN and environment explain matching physical outcomes, not equivalent learned parameters.\n')
