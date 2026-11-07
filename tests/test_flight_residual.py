import copy
import io
import unittest
from dataclasses import replace
import numpy as np
import torch
from Our_experiment.HCSAC.flight_residual.policy import (ACTION_DIM, INACTIVE_ACTION, VARIANTS, FlightMemory,
    flight_observe, local_features_and_probs)
from Our_experiment.HCSAC.flight_residual.critics import FlightCriticHead
from Our_experiment.HCSAC.flight_residual.trainer import TeamSAC
from Our_experiment.HCSAC.flight_residual.collector import collect_episode
from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import new_env
from Our_experiment.HCSAC.training.checkpointing import module_hash

SEEDS={'wind_seed':1180,'terrain_seed':900000,'infra_seed':900000}


class FlightResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(3001)
        cls.cnn=frozen_flight_actor();cls.env=new_env(80,SEEDS)
        cls.snapshot=flight_observe(cls.env,cls.env._get_obs(),cls.cnn)
        cls.agent=TeamSAC(VARIANTS[-1])
        cls.episode=collect_episode(cls.agent.actor,cls.cnn,SEEDS,75,random_actions=True)

    def test_cnn_feature_extraction_and_zero_residual_all_four(self):
        state=torch.as_tensor(self.env._get_obs(),dtype=torch.float32);mask=self.snapshot.action_mask
        z,p=local_features_and_probs(self.cnn,state,mask)
        with torch.no_grad():original=self.cnn(state[mask])
        self.assertTrue(torch.equal(p[mask],original));self.assertEqual(tuple(z.shape),(6,128))
        for variant in VARIANTS:
            torch.manual_seed(3001);actor=TeamSAC(variant).actor;memory=FlightMemory()
            with torch.no_grad():h=memory.advance(actor.encoder,self.snapshot);probs=actor.probabilities(h,self.snapshot)
            self.assertEqual(tuple(probs.shape),(6,8))
            self.assertTrue(torch.allclose(probs[mask],original,atol=1e-7,rtol=1e-6),variant)
            self.assertEqual(float(actor.residual_head[-1].weight.abs().max()),0.)

    def test_graph_feature_sensitivity_with_nonzero_head(self):
        torch.manual_seed(9);actor=TeamSAC(VARIANTS[2]).actor
        with torch.no_grad():actor.residual_head[-1].weight.normal_(std=.1)
        x=self.snapshot.x.clone();x[6,0]+=5
        altered=replace(self.snapshot,x=x)
        with torch.no_grad():
            p=actor.probabilities(FlightMemory().advance(actor.encoder,self.snapshot),self.snapshot)
            q=actor.probabilities(FlightMemory().advance(actor.encoder,altered),altered)
        self.assertFalse(torch.allclose(p[self.snapshot.action_mask],q[self.snapshot.action_mask],atol=1e-7))

    def test_eight_action_critic_and_inactive_sentinel(self):
        head=FlightCriticHead(8,True);self.assertEqual(head.action_embedding.num_embeddings,9)
        h=torch.randn(10,8);mask=torch.tensor([1,0,0,0,0,0],dtype=torch.bool)
        a=torch.tensor([7,0,0,0,0,0]);b=torch.tensor([7,7,7,7,7,7])
        self.assertTrue(torch.equal(head(h,a,mask)[0],head(h,b,mask)[0]))
        with self.assertRaises(ValueError):head(h,torch.tensor([8,0,0,0,0,0]),mask)
        self.assertEqual((ACTION_DIM,INACTIVE_ACTION),(8,8))

    def test_shared_initial_lamp_encoder_weights(self):
        models=[]
        for v in VARIANTS[1:]:torch.manual_seed(3001);models.append(TeamSAC(v))
        for component in ('actor','q1','q2','tq1','tq2'):
            ref=getattr(models[0],component).encoder.state_dict()
            for model in models[1:]:
                other=getattr(model,component).encoder.state_dict()
                self.assertTrue(all(torch.equal(x,other[k]) for k,x in ref.items()))

    def test_gradient_and_frozen_cnn(self):
        before_cnn=module_hash(self.cnn);agent=TeamSAC(VARIANTS[2]);before=copy.deepcopy(agent.actor.state_dict())
        tr=self.episode.transitions;agent.update([(tr[:4],tr[4:7])]);second=agent.update([(tr[:5],tr[5:8])])
        after=agent.actor.state_dict()
        self.assertGreater((before['residual_head.2.weight']-after['residual_head.2.weight']).norm().item(),0)
        self.assertGreater((before['encoder.U']-after['encoder.U']).norm().item(),0)
        self.assertGreater((before['encoder.conditioner.2.weight']-after['encoder.conditioner.2.weight']).norm().item(),0)
        self.assertGreater(second['conditioner_gradient_norm'],0);self.assertEqual(before_cnn,module_hash(self.cnn))

    def test_ecc_event_supervision_updates_both_critics(self):
        agent=TeamSAC(VARIANTS[-1]); ts=self.episode.transitions
        before=[copy.deepcopy(q.head.state_dict()) for q in (agent.q1,agent.q2)]
        result=agent.update([(ts[:3],ts[3:7])])
        self.assertTrue(np.isfinite(result['event_ce']))
        self.assertGreater(sum(result['event_counts']),0)
        for prior,q in zip(before,(agent.q1,agent.q2)):
            self.assertTrue(any(not torch.equal(value,q.head.state_dict()[key]) for key,value in prior.items()))

    def test_pre_flight_mask_and_post_flight_mask_separate(self):
        env=new_env(3,SEEDS);u=env.uavs[0];pos=tuple(u['position']);u['energy']=1000.
        ep=collect_episode(None,self.cnn,SEEDS,3,greedy=True,env_override=env);t=ep.transitions[0];row=ep.actions[0]
        self.assertTrue(t['snapshot'].action_mask[0]);self.assertFalse(t['offload_mask'][0])
        self.assertIsNotNone(row['proposed_action'][0]);self.assertEqual(row['execution_type'][0],'rule_return')
        self.assertEqual(row['override_reason'][0],'safety_energy_boundary')
        self.assertEqual(int(t['actions'][0]),row['proposed_action'][0]);self.assertTrue(t['event']&1)

    def test_nonactive_slots_do_not_propose_or_offload(self):
        for row in self.episode.actions:
            for i in range(6):
                if not row['flight_action_mask'][i]:self.assertIsNone(row['proposed_action'][i])
                if row['offload_action_mask'][i]:self.assertTrue(row['flight_action_mask'][i])
            self.assertEqual(sum(row['requested_offload']),sum(row['offload_action_mask']))
            self.assertEqual(sum(row['requested_offload'])-row['requested_offload'][2],0)
        self.assertEqual(self.episode.metrics['inactive_uav_offload_count'],0)

    def test_decision_sequence_and_reward_conservation(self):
        ts=self.episode.transitions;self.assertEqual(len(ts),75);self.assertTrue(ts[-1]['terminal'])
        for a,b in zip(ts,ts[1:]):
            self.assertIs(a['next_snapshot'],b['snapshot']);self.assertEqual(a['snapshot'].snapshot_id+1,b['snapshot'].snapshot_id)
        for t in ts:self.assertAlmostEqual(t['reward'],t['fly_reward']+t['offload_reward'],places=8)
        self.assertAlmostEqual(sum(t['reward'] for t in ts),self.episode.metrics['total_team_reward'],places=6)

    def test_memory_advances_once_and_events_not_inputs(self):
        actor=self.agent.actor;one=self.episode.transitions[0];other=dict(one);other['event']=7-one['event']
        m=FlightMemory()
        with torch.no_grad():
            h=m.advance(actor.encoder,one['snapshot']);same=m.advance(actor.encoder,one['snapshot'])
            p=actor.probabilities(h,one['snapshot']);q=actor.probabilities(same,other['snapshot'])
        self.assertTrue(torch.equal(h,same));self.assertTrue(torch.equal(p,q));m.reset();self.assertIsNone(m.hidden)

    def test_resume_reconstructs_same_random_update(self):
        agent=TeamSAC(VARIANTS[-1]);ts=self.episode.transitions;agent.update([(ts[:2],ts[2:4])])
        f=io.BytesIO();torch.save({'agent':agent.state_dict(),'replay':ts,'rng':torch.get_rng_state()},f)
        agent.update([(ts[:3],ts[3:5])]);expected=copy.deepcopy(agent.actor.state_dict())
        f.seek(0);saved=torch.load(f,weights_only=False);restored=TeamSAC(VARIANTS[-1]);restored.load_state_dict(saved['agent']);torch.set_rng_state(saved['rng'])
        restored.update([(saved['replay'][:3],saved['replay'][3:5])])
        self.assertEqual(restored.updates,2)
        self.assertTrue(all(torch.equal(expected[k],v) for k,v in restored.actor.state_dict().items()))

    def test_no_active_nonterminal_keeps_critic_bootstrap(self):
        agent=TeamSAC(VARIANTS[0]);t=dict(self.episode.transitions[2]);mask=torch.zeros(6,dtype=torch.bool)
        t['snapshot']=replace(t['snapshot'],action_mask=mask)
        t['next_snapshot']=replace(t['next_snapshot'],action_mask=mask)
        t['reward']=1.;t['terminal']=False
        with torch.no_grad():
            for q in (agent.q1,agent.q2):q.head.readout[-1].weight.zero_();q.head.readout[-1].bias.zero_()
            for q in (agent.tq1,agent.tq2):q.head.readout[-1].weight.zero_();q.head.readout[-1].bias.fill_(5.)
        result=agent.update([([], [t], torch.tensor([True]))])
        self.assertAlmostEqual(result['critic_loss'],(1+.99*5)**2,places=4);self.assertIsNone(result['actor_loss'])


if __name__=='__main__':unittest.main()
