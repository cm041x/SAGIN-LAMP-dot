import copy
import io
import random
import unittest
from dataclasses import replace
import torch
from Our_experiment.HCSAC.lamp_method.collector import collect_episode, frozen_flight_actor
from Our_experiment.HCSAC.lamp_method.policy import Actor, SnapshotMemory
from Our_experiment.HCSAC.lamp_method.trainer import TeamSAC
from Our_experiment.HCSAC.lamp_method.sequence_replay import SequenceReplay
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import new_env


class IntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(1)
        cls.agent=TeamSAC('lamp_lifecycle_ecc')
        cls.episode=collect_episode(cls.agent.actor,frozen_flight_actor(),
            {'wind_seed':1180,'terrain_seed':900000,'infra_seed':900000},70,random_actions=True)

    def test_decision_to_next_decision_and_reward_conservation(self):
        ts=self.episode.transitions
        self.assertEqual(len(ts),70);self.assertTrue(ts[-1]['terminal'])
        for a,b in zip(ts,ts[1:]):
            self.assertIs(a['next_snapshot'],b['snapshot'])
            self.assertEqual(a['snapshot'].snapshot_id+1,b['snapshot'].snapshot_id)
            self.assertFalse(a['terminal'])
        self.assertAlmostEqual(self.episode.prefix_flight_reward+sum(t['reward'] for t in ts),self.episode.total_physical_reward,places=6)
        self.assertGreater(self.episode.metrics['returns'],0)
        self.assertTrue(any(t['event'] & 1 for t in ts))

    def test_shared_lamp_encoders_initialize_identically_across_ablations(self):
        models=[]
        for variant in ('lamp_plain','lamp_lifecycle','lamp_lifecycle_ecc'):
            torch.manual_seed(3001);models.append(TeamSAC(variant))
        for network in ('actor','q1','q2','tq1','tq2'):
            reference=getattr(models[0],network).encoder.state_dict()
            for model in models[1:]:
                other=getattr(model,network).encoder.state_dict()
                self.assertTrue(all(torch.equal(v,other[k]) for k,v in reference.items()),network)

    def test_lifecycle_context_and_eligible_mask(self):
        snapshots=[t['snapshot'] for t in self.episode.transitions]
        self.assertTrue(torch.equal(snapshots[0].action_mask,torch.tensor([1,1,1,1,0,0],dtype=torch.bool)))
        self.assertEqual(snapshots[0].context[4,6].item(),1.)
        self.assertEqual(snapshots[0].context[4,11].item(),1.)
        self.assertTrue(any(s.context[:6,8].any() for s in snapshots))

    def test_snapshot_memory_does_not_double_advance_and_reset(self):
        m=SnapshotMemory();snap=self.episode.transitions[0]['snapshot']
        with torch.no_grad():
            first=m.advance(self.agent.actor.encoder,snap); second=m.advance(self.agent.actor.encoder,snap)
            self.assertTrue(torch.equal(first,second));m.reset()
            self.assertIsNone(m.hidden);self.assertTrue(torch.equal(first,m.advance(self.agent.actor.encoder,snap)))

    def test_different_memories_do_not_cross(self):
        a=SnapshotMemory();b=SnapshotMemory();ts=self.episode.transitions
        with torch.no_grad():
            h=a.advance(self.agent.actor.encoder,ts[0]['snapshot']);a.advance(self.agent.actor.encoder,ts[1]['snapshot'])
            self.assertTrue(torch.equal(h,b.advance(self.agent.actor.encoder,ts[0]['snapshot'])))

    def test_future_event_does_not_change_actor_output(self):
        transition=self.episode.transitions[10];changed=dict(transition);changed['event']=7-transition['event']
        with torch.no_grad():
            m1=SnapshotMemory();m2=SnapshotMemory()
            p1=self.agent.actor.probabilities(m1.advance(self.agent.actor.encoder,transition['snapshot']))
            p2=self.agent.actor.probabilities(m2.advance(self.agent.actor.encoder,changed['snapshot']))
        self.assertTrue(torch.equal(p1,p2))

    def test_ecc_and_lifecycle_parameters_update(self):
        agent=TeamSAC('lamp_lifecycle_ecc');before=copy.deepcopy(agent.actor.state_dict())
        before_q=copy.deepcopy(agent.q1.state_dict())
        ts=self.episode.transitions;out=agent.update([(ts[:5],ts[5:8])])
        self.assertTrue(torch.isfinite(torch.tensor(out['critic_loss'])))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in agent.actor.state_dict().items()))
        self.assertTrue(any(not torch.equal(before_q[k],v) for k,v in agent.q1.state_dict().items()))
        self.assertFalse(torch.equal(before['encoder.conditioner.2.weight'],agent.actor.state_dict()['encoder.conditioner.2.weight']))

    def test_no_actions_skips_actor_not_critic(self):
        agent=TeamSAC('lamp_lifecycle_ecc');t=dict(self.episode.transitions[-1])
        t['snapshot']=replace(t['snapshot'],action_mask=torch.zeros(6,dtype=torch.bool))
        before=copy.deepcopy(agent.actor.state_dict());qb=copy.deepcopy(agent.q1.state_dict())
        result=agent.update([([], [t])])
        self.assertIsNone(result['actor_loss'])
        self.assertTrue(all(torch.equal(before[k],v) for k,v in agent.actor.state_dict().items()))
        self.assertTrue(any(not torch.equal(qb[k],v) for k,v in agent.q1.state_dict().items()))

    def test_nonterminal_no_action_bootstraps_team_value(self):
        agent=TeamSAC('gcn_context');t=dict(self.episode.transitions[2])
        mask=torch.zeros(6,dtype=torch.bool)
        t['snapshot']=replace(t['snapshot'],action_mask=mask)
        t['next_snapshot']=replace(t['next_snapshot'],action_mask=mask)
        t['reward']=1.0;t['terminal']=False
        with torch.no_grad():
            for q in (agent.q1,agent.q2):
                q.head.readout[-1].weight.zero_();q.head.readout[-1].bias.zero_()
            for q in (agent.tq1,agent.tq2):
                q.head.readout[-1].weight.zero_();q.head.readout[-1].bias.fill_(5.)
        result=agent.update([([], [t], torch.tensor([True]))])
        self.assertAlmostEqual(result['critic_loss'],(1+.99*5)**2,places=4)
        self.assertIsNone(result['actor_loss'])

    def test_checkpoint_round_trip_preserves_actor_critics_and_replay(self):
        replay=SequenceReplay();replay.add_episode(self.episode.transitions)
        agent=TeamSAC('lamp_lifecycle_ecc'); agent.update([(self.episode.transitions[:2],self.episode.transitions[2:4])])
        buffer=io.BytesIO();torch.save({'agent':agent.state_dict(),'replay':replay.state_dict(),'torch_rng':torch.get_rng_state()},buffer)
        buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=False)
        restored=TeamSAC('lamp_lifecycle_ecc');restored.load_state_dict(saved['agent'])
        restored_replay=SequenceReplay();restored_replay.load_state_dict(saved['replay'])
        self.assertEqual(restored.updates,1);self.assertEqual(len(restored_replay.episodes[0]),70)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(agent.actor.parameters(),restored.actor.parameters())))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(agent.tq1.parameters(),restored.tq1.parameters())))

    def test_resume_reproduces_next_stochastic_update(self):
        agent=TeamSAC('lamp_lifecycle_ecc');seq=self.episode.transitions
        agent.update([(seq[:2],seq[2:5])])
        buff=io.BytesIO();torch.save({'agent':agent.state_dict(),'torch_rng':torch.get_rng_state()},buff)
        agent.update([(seq[:3],seq[3:6])]); expected=copy.deepcopy(agent.actor.state_dict())
        buff.seek(0);saved=torch.load(buff,weights_only=False);other=TeamSAC('lamp_lifecycle_ecc')
        other.load_state_dict(saved['agent']);torch.set_rng_state(saved['torch_rng'])
        other.update([(seq[:3],seq[3:6])])
        self.assertEqual(other.updates,2)
        self.assertTrue(all(torch.equal(expected[k],v) for k,v in other.actor.state_dict().items()))

    def test_padding_suffix_does_not_enter_loss_or_memory(self):
        torch.manual_seed(23);first=TeamSAC('lamp_lifecycle_ecc')
        second=copy.deepcopy(first);ts=self.episode.transitions
        seq=ts[2:4];padded=seq+[ts[4]]
        rng=torch.get_rng_state();first.update([(ts[:2],seq,torch.tensor([True,True]))])
        torch.set_rng_state(rng);second.update([(ts[:2],padded,torch.tensor([True,True,False]))])
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(first.actor.parameters(),second.actor.parameters())))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(first.q1.parameters(),second.q1.parameters())))

    def test_return_charge_and_reactivation_preserve_memory(self):
        agent=TeamSAC('lamp_lifecycle')
        ep=collect_episode(agent.actor,frozen_flight_actor(),
            {'wind_seed':1180,'terrain_seed':900000,'infra_seed':900000},180,greedy=True)
        memory=SnapshotMemory(); transitions=ep.transitions; transitions_seen=set()
        with torch.no_grad():
            previous=None
            for transition in transitions:
                snapshot=transition['snapshot'];current=snapshot.context[:6,6:11].argmax(-1)
                if previous is not None:
                    for old,new in zip(previous.tolist(),current.tolist()):
                        if old!=new:transitions_seen.add((old,new))
                memory.advance(agent.actor.encoder,snapshot)
                self.assertIsNotNone(memory.hidden)
                self.assertTrue(torch.isfinite(memory.hidden).all())
                previous=current
        self.assertIn((1,2),transitions_seen)  # active -> returning
        self.assertIn((4,0),transitions_seen)  # charging -> standby
        self.assertIn((0,1),transitions_seen)  # standby -> active

    def test_inherited_graph_task_feature_remains_at_reset_value(self):
        env=new_env(10,{'wind_seed':1180,'terrain_seed':900000,'infra_seed':900000})
        pos=tuple(env.uavs[0]['position']); initial=env.task_matrix[pos]
        occupied={tuple(u['position']) for u in env.uavs[1:4]}
        candidates=[i for i,d in enumerate(env.directions)
                    if 0<=pos[0]+d[0]<env.Lx and 0<=pos[1]+d[1]<env.Ly
                    and (pos[0]+d[0],pos[1]+d[1]) not in occupied
                    and env.task_matrix[pos[0]+d[0],pos[1]+d[1]]!=initial]
        self.assertTrue(candidates)
        actions=[0]*6;actions[0]=candidates[0];env.step(actions)
        current=env.task_matrix[tuple(env.uavs[0]['position'])]
        self.assertNotEqual(current,initial)
        self.assertEqual(env.get_obs_2().x[0,0].item(),initial)


if __name__=='__main__': unittest.main()
