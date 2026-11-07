"""Compare actual conditioner parameters to their deterministic initial state."""
import json
from pathlib import Path
import torch
from Our_experiment.HCSAC.lamp_method.trainer import TeamSAC

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/lamp_prototype'
torch.set_num_threads(2);results={}
for variant in ('lamp_lifecycle','lamp_lifecycle_ecc'):
    torch.manual_seed(3001);initial=TeamSAC(variant)
    ck=torch.load(OUT/'training'/variant/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    delta={}
    for network in ('actor','q1','q2'):
        before=getattr(initial,network).encoder.conditioner[-1].weight.detach()
        after=ck['agent'][network]['encoder.conditioner.2.weight']
        delta[network]=float((after-before).norm())
    results[variant]=delta
    if not all(x>0 for x in delta.values()):raise RuntimeError(f'Unchanged conditioner: {variant}')
(OUT/'conditioner_weight_audit.json').write_text(json.dumps({'passed':True,'delta_norms':results},indent=2)+'\n')
with (OUT/'PROTOTYPE_REPORT.md').open('a') as f:
    f.write('\nConditioner final-layer weight delta norms from deterministic initialization:\n\n')
    for variant,delta in results.items():f.write(f'- {variant}: {delta}.\n')
print('Conditioner actor and both critic weights changed:',results)
