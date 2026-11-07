"""Post-training audit and descriptive training statistics from saved artifacts."""
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from Our_experiment.HCSAC.lamp_method.trainer import TeamSAC

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/lamp_prototype'
VARIANTS=('gcn_context','lamp_plain','lamp_lifecycle','lamp_lifecycle_ecc')
torch.set_num_threads(2)


def mean_present(values):
    numbers=[float(x) for x in values if x is not None]
    return float(np.mean(numbers)) if numbers else None


def changed(initial,final):
    return sum(not torch.equal(initial[k],v) for k,v in final.items())


def event_weighted_mean(updates,key):
    observed=[(u[key],sum(u['event_counts'])) for u in updates if u[key] is not None]
    total=sum(n for _,n in observed)
    return sum(float(v)*n for v,n in observed)/total if total else None


rows=list(csv.DictReader((OUT/'per_run_metrics.csv').open()))
errors=[]; stats={}
if len(rows)!=36:errors.append(f'expected 36 evaluation rows, observed {len(rows)}')
keys={(r['variant'],r['wind_seed'],r['eval_seed']) for r in rows}
if len(keys)!=len(rows):errors.append('duplicate evaluation configurations')
for row in rows:
    if int(row['inactive_offload_count']) or int(row['energy_exhausted']):errors.append(f"unsafe evaluation: {row['variant']} {row['wind_seed']} {row['eval_seed']}")
for variant in VARIANTS:
    history=json.loads((OUT/'training'/variant/'history.json').read_text())
    ck=torch.load(OUT/'training'/variant/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    random.seed(3001);np.random.seed(3001);torch.manual_seed(3001);initial=TeamSAC(variant)
    updates=[u for e in history for u in e['updates']]
    counts=[sum(u['event_counts'][i] for u in updates) for i in range(8)]
    collected=[sum(t['event']==i for episode in ck['replay']['episodes'] for t in episode) for i in range(8)]
    majority=max(counts)/sum(counts) if sum(counts) else None
    entry={'episode':ck['episode'],'environment_steps':sum(e['steps'] for e in history),'updates':ck['agent']['updates'],
           'actor_parameter_tensors_changed':changed(initial.actor.state_dict(),ck['agent']['actor']),
           'critic1_parameter_tensors_changed':changed(initial.q1.state_dict(),ck['agent']['q1']),
           'critic2_parameter_tensors_changed':changed(initial.q2.state_dict(),ck['agent']['q2']),
           'actor_loss_mean':mean_present(u['actor_loss'] for u in updates),
           'critic_loss_mean':mean_present(u['critic_loss'] for u in updates),
           'event_ce_mean':mean_present(u['event_ce'] for u in updates),
           'event_brier_mean':mean_present(u['event_brier'] for u in updates),
           'event_accuracy_mean':mean_present(u['event_accuracy'] for u in updates),
           'event_ce_weighted':event_weighted_mean(updates,'event_ce'),
           'event_brier_weighted':event_weighted_mean(updates,'event_brier'),
           'event_accuracy_weighted':event_weighted_mean(updates,'event_accuracy'),
           'event_counts_both_critics':counts,'majority_class_accuracy_reference':majority,
           'event_counts_collected':collected,
           'majority_one_hot_brier_reference':2*(1-majority) if majority is not None else None,
           'conditioner_gradient_mean':mean_present(u['conditioner_gradient_norm'] for u in updates),
           'peak_rss_bytes':max(e['peak_rss_bytes'] for e in history),
           'seconds':sum(e['seconds'] for e in history)}
    if entry['episode']!=20 or entry['updates']!=360 or entry['environment_steps']!=4000:
        errors.append(f'{variant}: incomplete pilot')
    if not all(entry[k]>0 for k in ('actor_parameter_tensors_changed','critic1_parameter_tensors_changed','critic2_parameter_tensors_changed')):
        errors.append(f'{variant}: unchanged actor or critic')
    if variant.startswith('lamp_lifecycle') and not (entry['conditioner_gradient_mean'] and entry['conditioner_gradient_mean']>0):
        errors.append(f'{variant}: no conditioner gradient')
    stats[variant]=entry
audit={'passed':not errors,'errors':errors,'evaluation_count':len(rows),'training':stats,
       'limits':['event scores from training samples; no independent probability-calibration claim',
                 'single training seed per variant; do not infer robust superiority',
                 'peak RSS is the cumulative process high-water mark, not independent per-variant memory']}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
with (OUT/'PROTOTYPE_REPORT.md').open('a') as f:
    f.write('\n## Saved-checkpoint audit\n\n')
    f.write('Pilot status: '+('passed' if not errors else 'failed')+'; details in `audit.json`.\n\n')
    f.write('Classifier event counts above apply to ECC training batches; regular critics have no event head. '
            'The following counts are actual collected transition labels, including ordinary-critic variants. '
            'Weighted event scores use the same sampled-event counts as the majority reference.\n\n')
    for variant,s in stats.items():
        f.write(f"- {variant}: mean actor loss {s['actor_loss_mean']}, critic loss {s['critic_loss_mean']}; "
                f"changed actor/critic1/critic2 tensors {s['actor_parameter_tensors_changed']}/"
                f"{s['critic1_parameter_tensors_changed']}/{s['critic2_parameter_tensors_changed']}; "
                f"conditioner gradient {s['conditioner_gradient_mean']}; weighted event CE {s['event_ce_weighted']}; "
                f"weighted event accuracy {s['event_accuracy_weighted']}; weighted event Brier {s['event_brier_weighted']}; "
                f"majority one-hot Brier reference {s['majority_one_hot_brier_reference']}; "
                f"collected event counts {s['event_counts_collected']}.\n")
if errors:raise SystemExit('\n'.join(errors))
print('Pilot audit passed for',len(rows),'evaluation runs')
