import copy
import unittest
import torch
from Our_experiment.HCSAC.flight_stability.policy import VARIANTS,StableActor,FlightMemory,encode
from Our_experiment.HCSAC.flight_stability.trainer import TeamSAC
from Our_experiment.HCSAC.flight_stability.collector import collect_episode
from Our_experiment.HCSAC.lamp_method.collector import frozen_flight_actor

SEEDS={'wind_seed':1180,'terrain_seed':900000,'infra_seed':900000}

class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.cnn=frozen_flight_actor()
        cls.ep=collect_episode(None,cls.cnn,SEEDS,30,greedy=True)
        cls.probe={'snapshots':[x['snapshot'] for x in cls.ep.transitions[:16]],'indices':[0,5,10,15]}

    def test_zero_residual_and_bounded_logits(self):
        s=self.ep.transitions[0]['snapshot']
        for variant in VARIANTS:
            actor=StableActor(variant);h=FlightMemory().advance(actor.encoder,s)
            self.assertTrue(torch.allclose(actor.probabilities(h,s)[s.action_mask],s.base_probs[s.action_mask],atol=1e-7))
            with torch.no_grad():actor.residual_head[-1].bias.fill_(20)
            self.assertLessEqual(float(actor.residual(h,s).abs().max()),1.)
            actor.bounded=False
            self.assertGreater(float(actor.residual(h,s).abs().max()),1.)

    def test_raw_memory_not_readout_and_recurrence(self):
        actor=StableActor(VARIANTS[1]);m=FlightMemory();s=self.ep.transitions[0]['snapshot']
        h=m.advance(actor.encoder,s)
        self.assertIsNotNone(m.hidden);self.assertFalse(torch.equal(h,m.hidden))
        self.assertTrue(torch.equal(h,m.advance(actor.encoder,s)))
        h2=m.advance(actor.encoder,self.ep.transitions[1]['snapshot']);self.assertTrue(torch.isfinite(h2).all())
        self.assertEqual(actor.encoder.core.xi,2.)

    def test_kl_rejection_restores_actor_and_optimizer(self):
        agent=TeamSAC(VARIANTS[0]);agent.actor_optimizer.zero_grad()
        for p in agent.actor.parameters():p.grad=torch.ones_like(p)*100
        before=copy.deepcopy(agent.actor.state_dict());opt=copy.deepcopy(agent.actor_optimizer.state_dict())
        outcome=agent.guarded_actor_step(self.probe,max_kl=-1,max_attempts=3)
        self.assertFalse(outcome['accepted']);self.assertEqual(outcome['attempts'],3)
        self.assertTrue(all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in before.items()))
        self.assertEqual(opt,agent.actor_optimizer.state_dict())

    def test_warmstart_critic_only_and_guarded_actor(self):
        a=TeamSAC(VARIANTS[0]);ts=self.ep.transitions;actor=copy.deepcopy(a.actor.state_dict());q=copy.deepcopy(a.q1.state_dict())
        out=a.update([(ts[:2],ts[2:5])],actor_enabled=False)
        self.assertIsNone(out['actor_loss']);self.assertTrue(all(torch.equal(v,a.actor.state_dict()[k]) for k,v in actor.items()))
        self.assertTrue(any(not torch.equal(v,a.q1.state_dict()[k]) for k,v in q.items()))
        out=a.update([(ts[:2],ts[2:5])],probe=self.probe)
        self.assertIsNotNone(out['actor_guard']);self.assertLessEqual(out['actor_guard']['kl'],.01)

    def test_rewards_match_components(self):
        m=self.ep.metrics
        self.assertAlmostEqual(m['total_team_reward'],m['coverage_gain_reward']+m['first_visit_bonus']-m['block_penalty']-m['search_energy_penalty']+m['total_offload_reward'],places=5)
        self.assertEqual(m['inactive_uav_offload_count'],0)

if __name__=='__main__':unittest.main()
