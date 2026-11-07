import hashlib
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT/'Our_experiment/HCSAC');torch.set_num_threads(2)
from Our_experiment.HCSAC.training.masked_discrete_sac import MaskedDiscreteSAC
from Our_experiment.HCSAC.training.replay_buffers import CNNReplay,GraphReplay
from Our_experiment.HCSAC.training.checkpointing import agent_state,restore_agent,atomic_save,module_hash,rng_state,restore_rng,file_hash
from Our_experiment.HCSAC.training.training_split import build_split,episode_manifest
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import TrainingMode,initialize_agents,new_env,collect_step,choose_flight
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import Status

class DynamicTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=new_env(10,{'wind_seed':11,'terrain_seed':42,'infra_seed':42})
        cls.graph=cls.env.get_obs_2();cls.state=cls.env._get_obs()
    def graph_samples(self, active=None, next_active=None):
        b=GraphReplay(4);active=active or [True]*4+[False]*2;next_active=next_active or active
        b.add(self.graph,[0]*6,[.1]*6,self.graph,active,next_active,[False]*6,False,1,1)
        return b.sample(1)
    def cnn_samples(self):
        b=CNNReplay(4);b.add(self.state[0],0,.1,self.state[0],False,0,1,1);return b.sample(1)
    def test_training_modes_are_distinct(self):self.assertNotEqual(TrainingMode.SCRATCH,TrainingMode.FINETUNE)
    def test_scratch_initialization_differs_from_original(self):
        a,_=initialize_agents(TrainingMode.SCRATCH,'cpu');b,_=initialize_agents(TrainingMode.FINETUNE,'cpu');self.assertNotEqual(module_hash(a.actor),module_hash(b.actor))
    def test_finetune_initialization_matches_original_actor(self):
        a,b=initialize_agents(TrainingMode.FINETUNE,'cpu');c=MaskedDiscreteSAC('CNN');c.actor.load_state_dict(torch.load(ROOT/'Our_experiment/HCSAC/data/sac_model_fly.pt',map_location='cpu',weights_only=True)['actor_state_dict'])
        self.assertEqual(module_hash(a.actor),module_hash(c.actor))
        with torch.no_grad():self.assertTrue(torch.equal(a.actor(torch.tensor(self.state[:1],dtype=torch.float32)),c.actor(torch.tensor(self.state[:1],dtype=torch.float32))))
        original=torch.load(ROOT/'Our_experiment/HCSAC/data/sac_model_offload.pt',map_location='cpu',weights_only=True)['actor_state_dict'];self.assertTrue(all(torch.equal(b.actor.state_dict()[k],v) for k,v in original.items()))
    def test_finetune_uses_fresh_optimizers(self):
        a,b=initialize_agents(TrainingMode.FINETUNE,'cpu');self.assertEqual(len(a.actor_optimizer.state),0);self.assertEqual(len(b.critic_1_optimizer.state),0)
    def test_cnn_buffer_only_stores_executed_active_slots(self):
        e=new_env(10,{'wind_seed':11,'terrain_seed':42,'infra_seed':42});a,b=initialize_agents(TrainingMode.FINETUNE,'cpu');c=CNNReplay(20);g=GraphReplay(20)
        before=e.get_search_active_mask();collect_step(e,e._get_obs(),a,b,c,g,1,1,greedy=True)
        self.assertEqual({x[5] for x in c.items},set(e.eligible));self.assertTrue(all(before[x[5]] for x in c.items))
    def test_newly_launched_uav_not_stored_before_execution(self):
        e=new_env(10,{'wind_seed':11,'terrain_seed':42,'infra_seed':42});e.uavs[0]['energy']=.44*e.initial_energy
        a,b=initialize_agents(TrainingMode.FINETUNE,'cpu');c=CNNReplay(20);g=GraphReplay(20);collect_step(e,e._get_obs(),a,b,c,g,1,1,greedy=True)
        self.assertTrue(any(u['status']==Status.ACTIVE for u in e.uavs[4:]));self.assertTrue(all(x[5]<4 for x in c.items))
    def test_returning_uav_not_stored(self):
        e=new_env(10,{'wind_seed':11,'terrain_seed':42,'infra_seed':42});e.uavs[0].update(status=Status.RETURNING,done=True)
        a,b=initialize_agents(TrainingMode.FINETUNE,'cpu');c=CNNReplay(20);g=GraphReplay(20);collect_step(e,e._get_obs(),a,b,c,g,1,1,greedy=True);self.assertNotIn(0,[x[5] for x in c.items])
    def test_standby_uav_not_stored(self):self.assertFalse(self.env.get_active_mask()[4])
    def test_charging_uav_not_stored(self):
        e=new_env(10,{'wind_seed':11,'terrain_seed':42,'infra_seed':42});e.uavs[4].update(status=Status.CHARGING,done=True,energy=100000)
        a,b=initialize_agents(TrainingMode.FINETUNE,'cpu');c=CNNReplay(20);g=GraphReplay(20);collect_step(e,e._get_obs(),a,b,c,g,1,1,greedy=True);self.assertNotIn(4,[x[5] for x in c.items])
    def test_cnn_slot_done_when_inactive_next_state(self):
        b=CNNReplay(1);b.add(self.state[0],0,0,np.zeros_like(self.state[0]),True,0,1,1);self.assertTrue(b.items[0][4])
    def test_graph_buffer_clones_data(self):
        b=GraphReplay(1);g=self.graph.clone();b.add(g,[0]*6,[0]*6,g,[1]*4+[0]*2,[1]*4+[0]*2,[0]*6,False,1,1);saved=b.items[0][0].x.clone();g.x[0,0]=999;self.assertTrue(torch.equal(b.items[0][0].x,saved))
    def test_graph_slot_mapping_after_batching(self):
        a=MaskedDiscreteSAC('GCN');out=a._graph(a.actor,[self.graph,self.graph]);self.assertEqual(tuple(out.shape),(2,6,5));self.assertTrue(torch.allclose(out[0],out[1]))
    def test_graph_mapping_is_independent_of_node_insertion_order(self):
        from torch_geometric.data import Data
        order=torch.tensor([3,0,5,2,1,4,9,7,6,8]);inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order))
        g=self.graph
        perm=Data(x=g.x[order],node_type=g.node_type[order],slot_id=g.slot_id[order],
                  edge_index=inverse[g.edge_index],edge_weight=g.edge_weight)
        a=MaskedDiscreteSAC('GCN')
        with torch.no_grad():
            expected=a._graph(a.actor,[g]);actual=a._graph(a.actor,[perm])
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6))

    def test_gcn_loss_ignores_inactive_slots(self):
        torch.manual_seed(9);a=MaskedDiscreteSAC('GCN');torch.manual_seed(9);b=MaskedDiscreteSAC('GCN');samples=self.graph_samples();mut=list(samples[0]);mut[1]=mut[1].copy();mut[2]=mut[2].copy();mut[1][4:]=4;mut[2][4:]=1e6
        x=a.update(samples);y=b.update([tuple(mut)]);self.assertAlmostEqual(x['critic_loss'],y['critic_loss'],places=6)
    def test_gcn_bootstrap_ignores_inactive_next_slots(self):
        torch.manual_seed(8);a=MaskedDiscreteSAC('GCN');torch.manual_seed(8);b=MaskedDiscreteSAC('GCN');s=self.graph_samples([False,False,False,False,True,False],[False]*6);t=list(s[0]);t[3]=t[3].clone();t[3].x[4]=100
        self.assertAlmostEqual(a.update(s)['critic_loss'],b.update([tuple(t)])['critic_loss'],places=6)
    def test_actor_loss_ignores_inactive_slots(self):
        torch.manual_seed(7);a=MaskedDiscreteSAC('GCN');torch.manual_seed(7);b=MaskedDiscreteSAC('GCN');s=self.graph_samples();t=list(s[0]);t[0]=t[0].clone();t[0].x[4]=torch.tensor([100.,100.,100.,100.,100.,100.])
        self.assertAlmostEqual(a.update(s)['actor_loss'],b.update([tuple(t)])['actor_loss'],places=6)
    def test_alpha_loss_ignores_inactive_slots(self):
        torch.manual_seed(6);a=MaskedDiscreteSAC('GCN');torch.manual_seed(6);b=MaskedDiscreteSAC('GCN');s=self.graph_samples();t=list(s[0]);t[0]=t[0].clone();t[0].x[4]=torch.tensor([100.,100.,100.,100.,100.,100.])
        self.assertAlmostEqual(a.update(s)['alpha_loss'],b.update([tuple(t)])['alpha_loss'],places=6)
    def test_empty_active_mask_update_is_safe(self):
        a=MaskedDiscreteSAC('GCN');s=self.graph_samples([False]*6);self.assertTrue(a.update(s)['skipped']);self.assertEqual(a.skipped_empty_mask_updates,1)
    def test_cnn_single_update_changes_weights(self):
        a=MaskedDiscreteSAC('CNN');old=module_hash(a.actor);a.update(self.cnn_samples());self.assertNotEqual(old,module_hash(a.actor))
    def test_gcn_single_update_changes_weights(self):
        a=MaskedDiscreteSAC('GCN');old=module_hash(a.actor);a.update(self.graph_samples());self.assertNotEqual(old,module_hash(a.actor))
    def test_losses_are_finite(self):
        for kind,samples in [('CNN',self.cnn_samples()),('GCN',self.graph_samples())]:
            r=MaskedDiscreteSAC(kind).update(samples);self.assertTrue(all(np.isfinite(r[k]) for k in ('actor_loss','critic_loss','alpha_loss','alpha','entropy')))
    def test_log_alpha_on_correct_device(self):self.assertEqual(MaskedDiscreteSAC('CNN').log_alpha.device.type,'cpu')
    def test_checkpoint_roundtrip(self):
        a=MaskedDiscreteSAC('GCN');a.update(self.graph_samples());b=MaskedDiscreteSAC('GCN')
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p=Path(d)/'c.pt';atomic_save(p,agent_state(a));restore_agent(b,torch.load(p,map_location='cpu',weights_only=False))
        self.assertEqual(module_hash(a.actor),module_hash(b.actor));self.assertTrue(torch.equal(a.log_alpha,b.log_alpha))
    def test_resume_restores_replay_buffers(self):
        a=GraphReplay(2);s=self.graph_samples()[0];a.items.append(s);b=GraphReplay(2);b.load_state_dict(a.state_dict());self.assertEqual(len(a),len(b));self.assertTrue(torch.equal(a.items[0][0].x,b.items[0][0].x))
    def test_resume_restores_rng_state(self):
        random.seed(3);np.random.seed(3);torch.manual_seed(3);saved=rng_state();v=(random.random(),np.random.rand(),torch.rand(1));restore_rng(saved);w=(random.random(),np.random.rand(),torch.rand(1));self.assertEqual(v[:2],w[:2]);self.assertTrue(torch.equal(v[2],w[2]))
    def test_validation_deterministic_argmax(self):
        a=MaskedDiscreteSAC('CNN');self.assertEqual(choose_flight(a,self.state,self.env.get_active_mask(),random_actions=False,greedy=True),choose_flight(a,self.state,self.env.get_active_mask(),random_actions=False,greedy=True))
    def test_training_and_test_splits_are_disjoint(self):
        s=build_split();tr={w for xs in s['train_wind_seed_pools'].values() for w in xs};va={x['wind_seed'] for x in s['validation_combinations']};te={x['wind_seed'] for x in s['test_combinations']};self.assertTrue(tr.isdisjoint(va|te));self.assertTrue(va.isdisjoint(te));self.assertEqual(len(va),9)
    def test_scratch_and_finetune_share_episode_manifest(self):
        s=build_split();self.assertEqual(episode_manifest(3001,20,s),episode_manifest(3001,20,s));self.assertEqual(episode_manifest(3001,10,s),episode_manifest(3001,15,s)[:10])
    def test_original_v15_environment_unchanged(self):
        p='Our_experiment/HCSAC/ENV/UAVenv_DynamicFleetV15.py'
        manifest=json.loads((ROOT/'SOURCE_SNAPSHOT_SHA256.json').read_text())
        hashes={entry['path']:entry['sha256'] for entry in manifest['files']}
        self.assertEqual(file_hash(ROOT/p),hashes[p])
    def test_original_pretrained_weights_unchanged(self):
        entries=dict(line.split('  ',1)[::-1] for line in (ROOT/'SOURCE_BASELINE_SHA256.txt').read_text().splitlines());p='Our_experiment/HCSAC/data/sac_model_fly.pt';self.assertEqual(file_hash(ROOT/p),entries[p])
    def test_zero_shot_checkpoint_not_modified(self):
        entries=dict(line.split('  ',1)[::-1] for line in (ROOT/'SOURCE_BASELINE_SHA256.txt').read_text().splitlines());p='Our_experiment/HCSAC/data/sac_model_offload.pt';self.assertEqual(file_hash(ROOT/p),entries[p])
if __name__=='__main__':unittest.main()
