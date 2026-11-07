"""Five zero-residual 500-step validation-scenario rollouts before training."""
import hashlib
import json
from pathlib import Path
import torch
from Our_experiment.HCSAC.flight_residual.policy import VARIANTS
from Our_experiment.HCSAC.flight_residual.trainer import TeamSAC
from Our_experiment.HCSAC.flight_residual.collector import collect_episode
from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/lamp_flight_residual_pilot'
torch.set_num_threads(2);cnn=frozen_flight_actor();results={};baseline=None
for v in ('frozen_cnn_reference',*VARIANTS):
    torch.manual_seed(3001);actor=None if v=='frozen_cnn_reference' else TeamSAC(v).actor
    episode=collect_episode(actor,cnn,{'wind_seed':1180,'terrain_seed':900000,'infra_seed':900000},500,greedy=True)
    digest=hashlib.sha256(json.dumps(episode.trajectory,sort_keys=True).encode()).hexdigest()
    if baseline is None:baseline=digest
    matches=digest==baseline
    if episode.metrics['steps']!=500 or episode.metrics['inactive_uav_offload_count'] or not matches:
        raise RuntimeError(f'Zero-residual smoke mismatch for {v}')
    results[v]={'trajectory_sha256':digest,'matches_reference':matches,'metrics':episode.metrics}
    print('flight smoke',v,'steps',episode.metrics['steps'],'overrides',episode.metrics['flight_overrides'],flush=True)
(OUT/'smoke_results.json').write_text(json.dumps({'passed':True,'variants':results},indent=2,allow_nan=False)+'\n')
