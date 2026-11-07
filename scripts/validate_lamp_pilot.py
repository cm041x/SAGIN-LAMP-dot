"""Evaluate final pilot actors on the shared nine validation scenarios."""
import hashlib
import json
import os
from pathlib import Path
import torch
from Our_experiment.HCSAC.training.training_split import build_split
from Our_experiment.HCSAC.lamp_method.trainer import TeamSAC
from Our_experiment.HCSAC.lamp_method.collector import collect_episode,frozen_flight_actor

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/lamp_prototype'
VARIANTS=('gcn_context','lamp_plain','lamp_lifecycle','lamp_lifecycle_ecc')
torch.set_num_threads(2)
flight=frozen_flight_actor(); scenarios=build_split()['validation_combinations']
for variant in VARIANTS:
    path=OUT/'training'/variant/'last_checkpoint.pt'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint['episode']!=20:raise RuntimeError(f'{variant} pilot is incomplete')
    agent=TeamSAC(variant);agent.load_state_dict(checkpoint['agent'])
    for scenario in scenarios:
        dest=OUT/'validation'/variant/f"w{scenario['wind_seed']}_t{scenario['terrain_seed']}.json"
        if dest.exists():
            if json.loads(dest.read_text())['checkpoint_sha256']!=digest:raise RuntimeError('stale validation output')
            continue
        episode=collect_episode(agent.actor,flight,scenario,500,greedy=True)
        row={'variant':variant,'checkpoint_sha256':digest,**scenario,**episode.metrics}
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(row,allow_nan=False)+'\n');os.replace(tmp,dest)
        print('validation',variant,scenario['wind_seed'],flush=True)
