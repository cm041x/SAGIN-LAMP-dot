"""Deterministic, bounded end-to-end diagnostic; not a research experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT / 'Our_experiment/HCSAC')
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import DynamicFleetConfig, DynamicFleetEnv, Status
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import PolicyNet_CNN, PolicyNet_GCN

def source_unchanged() -> bool:
    src=Path('/Volumes/czn/Paper/SAGIN/Multi-UAV-Search-SAGIN-HybridDRLGA-main')
    return all(hashlib.sha256((src/p).read_bytes()).hexdigest()==h for h,p in
               (line.split('  ',1) for line in (ROOT/'SOURCE_BASELINE_SHA256.txt').read_text().splitlines()))

def heuristic(env: DynamicFleetEnv) -> list[int]:
    result=[]
    for u in env.uavs:
        pos=u['position']; choices=[]
        for i,(dx,dy) in enumerate(env.directions):
            x,y=pos[0]+dx,pos[1]+dy
            if 0<=x<env.Lx and 0<=y<env.Ly:
                choices.append((env.uncertainty_matrix[x,y], -i, i))
        result.append(max(choices)[2] if choices else 0)
    return result

def run(seed: int, max_steps: int, output_dir: str, controller: str) -> dict:
    out=Path(output_dir)
    if not out.is_absolute(): out=ROOT/out
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    c=DynamicFleetConfig(seed=seed, max_episode_steps=max_steps, charging_vehicle_position=(10,10),
        charge_rate_per_step=30000, launch_trigger_ratio=0.8, return_trigger_ratio=0.6, handover_min_steps=2,
        enable_pretrained_controller=controller=='pretrained')
    env=DynamicFleetEnv(c)
    initial=[(7,10),(13,10),(10,7),(10,13),(10,10),(10,10)]
    state=env.reset(seed=seed, positions=initial, destinations=[c.charging_vehicle_position]*6)
    env.uavs[0]['energy']=env.initial_energy*0.57
    env._log_event('smoke_energy_override', 0, Status.ACTIVE, Status.ACTIVE, ratio=0.57)
    cnn=gcn=None
    if controller=='pretrained':
        cnn=PolicyNet_CNN(15,128,8).eval(); gcn=PolicyNet_GCN(6,128,5).eval()
        cnn.load_state_dict(torch.load('data/sac_model_fly.pt',map_location='cpu',weights_only=True)['actor_state_dict'])
        gcn.load_state_dict(torch.load('data/sac_model_offload.pt',map_location='cpu',weights_only=True)['actor_state_dict'])
    snapshots=[]
    def snapshot() -> dict:
        return {'step':env.step_number, **env.get_fleet_counts(), 'energy':[float(u['energy']) for u in env.uavs],
                'position':[list(u['position']) for u in env.uavs], 'statuses':env.get_uav_statuses(),
                'uncertainty':float(env.uncertainty_matrix.mean())}
    snapshots.append(snapshot())
    for _ in range(max_steps):
        if cnn is None:
            actions=heuristic(env)
        else:
            actions=[0]*env.N
            active_slots=np.flatnonzero(env.get_search_active_mask()).tolist()
            if active_slots:
                with torch.no_grad():
                    selected=cnn(torch.as_tensor(state[active_slots],dtype=torch.float32)).argmax(dim=1).tolist()
                for slot,action in zip(active_slots,selected):
                    actions[slot]=action
        old_uncertainty=env.uncertainty_matrix.copy()
        old_energy=[u['energy'] for u in env.uavs]
        old_status=env.get_uav_statuses()
        state, _, done=env.step(actions)
        assert np.isfinite(state).all()
        changed=set(map(tuple,np.argwhere(env.uncertainty_matrix != old_uncertainty)))
        searched={tuple(env.uavs[i]['position']) for i in env.eligible}
        assert changed <= searched
        for i, status in enumerate(old_status):
            if status == Status.CHARGING.value:
                assert env.uavs[i]['energy'] >= old_energy[i]
        graph=env.get_obs_2()
        assert torch.isfinite(graph.x).all() and torch.isfinite(graph.edge_weight).all()
        assert graph.slot_id[:6].tolist()==list(range(6))
        mask=env.get_search_active_mask()
        for i,active in enumerate(mask):
            if not active:
                assert not (graph.edge_index==i).any() and not graph.x[i].any()
        if gcn is None:
            offloads=[0]*6
        else:
            with torch.no_grad(): offloads=gcn(graph).argmax(dim=1).tolist()
        assert len(offloads)==6
        before=[u['energy'] for u in env.uavs]
        _, rewards, _=env.step_offload(offloads)
        for i in range(6):
            if i not in env.eligible:
                assert rewards[i]==0 and env.uavs[i]['energy']==before[i]
        for u in env.uavs:
            assert np.isfinite(u['energy']) and 0 <= u['energy'] <= env.initial_energy
        snapshots.append(snapshot())
        if done: break
    events=env.get_event_log()
    with (out/'events.jsonl').open('w') as f:
        for e in events: f.write(json.dumps(e,allow_nan=False)+'\n')
    (out/'snapshots.json').write_text(json.dumps(snapshots,indent=2,allow_nan=False)+'\n')
    types=[e['event_type'] for e in events]
    moves=[e for e in events if e['event_type']=='return_move']
    for e in moves:
        a=e['metadata']['from_position']; b=e['position']
        assert max(abs(a[i]-b[i]) for i in (0,1))==1
        assert max(abs(b[i]-10) for i in (0,1)) < max(abs(a[i]-10) for i in (0,1))
    counts=[s['search_active_count'] for s in snapshots]
    air=[s['airborne_count'] for s in snapshots]
    abnormalities=[e for e in events if e['event_type']=='energy_exhausted_before_return']
    full_ids={e['uav_id'] for e in events if e['event_type']=='charging_to_standby'}
    closed=any(e['event_type']=='standby_to_active' and e['uav_id'] in full_ids and
        any(f['event_type']=='charging_to_standby' and f['uav_id']==e['uav_id'] and f['step']<e['step'] for f in events) for e in events)
    passed=all(t in types for t in ('reserve_launched','active_to_returning','returned_to_vehicle','charging_started','charging_to_standby')) and bool(moves) and min(counts)<max(counts) and not abnormalities and closed
    fingerprint = hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in ('Our_experiment/HCSAC/ENV/UAVenv_DynamicFleet.py', 'Our_experiment/HCSAC/dynamic_fleet_smoke_test.py', 'Our_experiment/HCSAC/dynamic_fleet_pretrained_test.py', 'Our_experiment/HCSAC/data/sac_model_fly.pt', 'Our_experiment/HCSAC/data/sac_model_offload.pt'))).hexdigest()
    summary={'code_and_weights_sha256':fingerprint,'actual_project_path':str(ROOT),'seed':seed,'max_steps':max_steps,'initial_active_uavs':4,'reserve_uavs':2,
        'charging_vehicle_position':list(c.charging_vehicle_position),'charging_capacity':c.charging_capacity,
        'charge_rate_per_step':c.charge_rate_per_step,'launch_trigger_ratio':c.launch_trigger_ratio,'return_trigger_ratio':c.return_trigger_ratio,
        'total_launch_events':types.count('reserve_launched'),'total_return_events':types.count('active_to_returning'),
        'total_charging_start_events':types.count('charging_started'),'total_charging_complete_events':types.count('charging_completed'),
        'min_search_active_count':min(counts),'max_search_active_count':max(counts),'final_search_active_count':counts[-1],
        'min_airborne_count':min(air),'max_airborne_count':max(air),'final_average_uncertainty':snapshots[-1]['uncertainty'],
        'final_coverage':1-snapshots[-1]['uncertainty'],'final_uav_statuses':env.get_uav_statuses(),
        'per_uav_activation_count':[u['activation_count'] for u in env.uavs],
        'per_uav_charging_count':[u['charging_count'] for u in env.uavs], 'abnormal_events':abnormalities,
        'heuristic_smoke_passed':passed if controller=='heuristic' else None,
        'pretrained_smoke_passed':passed if controller=='pretrained' else None,
        'recharged_uav_reactivated':closed, 'original_source_unchanged':source_unchanged()}
    # Link results only when both runs used identical code, weights and settings.
    peer_name = 'dynamic_fleet_pretrained' if controller == 'heuristic' else 'dynamic_fleet_smoke'
    peer_path = out.parent / peer_name / 'summary.json'
    if peer_path.is_file():
        peer = json.loads(peer_path.read_text())
        same = peer.get('code_and_weights_sha256') == fingerprint and peer.get('seed') == seed and peer.get('max_steps') == max_steps
        if same:
            other_key = 'pretrained_smoke_passed' if controller == 'heuristic' else 'heuristic_smoke_passed'
            own_key = 'heuristic_smoke_passed' if controller == 'heuristic' else 'pretrained_smoke_passed'
            summary[other_key] = peer.get(other_key)
            peer[own_key] = summary[own_key]
            summary['other_test_summary'] = '../' + peer_name + '/summary.json'
            peer['other_test_summary'] = '../' + out.name + '/summary.json'
            peer_path.write_text(json.dumps(peer,indent=2,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    xs=[s['step'] for s in snapshots]
    def lineplot(filename: str, series: list[tuple[str,list]], ylabel: str) -> None:
        fig,ax=plt.subplots(figsize=(9,4))
        for label, values in series: ax.plot(xs,values,label=label)
        ax.set(xlabel='Step',ylabel=ylabel); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/filename,dpi=140); plt.close(fig)
    lineplot('fleet_timeline.png',[('Search active',counts),('Airborne',air)],'UAV count')
    lineplot('search_active_count.png',[('Search active',counts)],'UAV count')
    lineplot('airborne_count.png',[('Airborne',air)],'UAV count')
    lineplot('energy_curves.png',[(f'UAV {i}',[s['energy'][i] for s in snapshots]) for i in range(6)],'Energy (J)')
    lineplot('uncertainty_curve.png',[('Mean uncertainty',[s['uncertainty'] for s in snapshots])],'Uncertainty')
    labels=[s.value for s in Status]; matrix=np.array([[labels.index(s['statuses'][i]) for s in snapshots] for i in range(6)])
    fig,ax=plt.subplots(figsize=(10,3.5)); ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#999999','#4daf4a','#377eb8','#ffbf00','#e41a1c']),vmin=0,vmax=4)
    ax.set(xlabel='Step',ylabel='UAV slot',yticks=range(6)); ax.legend(handles=[plt.Line2D([0],[0],color=c,lw=5,label=l) for l,c in zip(labels,['#999999','#4daf4a','#377eb8','#ffbf00','#e41a1c'])],bbox_to_anchor=(1.01,1),loc='upper left'); fig.tight_layout(); fig.savefig(out/'uav_status_timeline.png',dpi=140); plt.close(fig)
    print(json.dumps(summary,indent=2)); return summary

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42); p.add_argument('--max-steps',type=int,default=200)
    p.add_argument('--output-dir',default='Our_experiment/HCSAC/data/dynamic_fleet_smoke'); p.add_argument('--controller',choices=['heuristic','pretrained'],default='heuristic')
    a=p.parse_args(); s=run(a.seed,a.max_steps,a.output_dir,a.controller)
    if not s['heuristic_smoke_passed' if a.controller=='heuristic' else 'pretrained_smoke_passed']: raise SystemExit('Smoke acceptance failed; see summary and events')
if __name__=='__main__': main()
