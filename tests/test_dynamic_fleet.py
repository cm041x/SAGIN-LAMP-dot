import os
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / 'Our_experiment/HCSAC')
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import DynamicFleetConfig, DynamicFleetEnv, Status
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import UAVEnv

class DynamicFleetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DynamicFleetEnv(DynamicFleetConfig(charging_vehicle_position=(10,10), max_episode_steps=100))

    def setUp(self):
        self.e = self.env
        self.e.reset(seed=42)

    def move_to_vehicle_zone(self, index=0):
        u=self.e.uavs[index]
        u['position']=(7,10)
        u['energy']=self.e.initial_energy*0.25
        u['status']=Status.RETURNING
        u['done']=True
        return u

    def test_reset_and_masks(self):
        self.assertEqual(len(self.e.uavs),6)
        self.assertEqual(self.e.get_uav_statuses(), ['active']*4+['standby']*2)
        self.assertEqual(self.e.get_active_mask().tolist(), [True]*4+[False]*2)
        self.assertEqual(self.e.active_mask.tolist(), [True]*4+[False]*2)
        self.assertEqual(len(self.e.get_airborne_mask()),6)

    def test_standby_stationary_and_no_energy(self):
        u=self.e.uavs[4]; before=(u['position'],u['energy'])
        self.e.step([0]*6); rewards=self.e.step_offload([0]*6)[1]
        self.assertEqual((u['position'],u['energy']),before)
        self.assertEqual(rewards[4:], [0,0])

    def test_standby_no_uncertainty_update(self):
        for u in self.e.uavs[:4]: u['status']=Status.STANDBY; u['done']=True
        before=self.e.uncertainty_matrix.copy()
        self.e.step([0]*6)
        np.testing.assert_array_equal(before,self.e.uncertainty_matrix)

    def test_single_replacement_request(self):
        self.e.uavs[0]['energy']=self.e.initial_energy*0.44
        self.e.step([0]*6)
        self.assertEqual(self.e.get_fleet_counts()['standby_count'],1)
        self.e.step([0]*6)
        self.assertEqual(sum(v['event_type']=='reserve_launched' and v['metadata']['replacement_for']==0 for v in self.e.events),1)

    def test_return_moves_toward_home_without_search(self):
        u=self.move_to_vehicle_zone(); before=self.e.uncertainty_matrix.copy()
        for v in self.e.uavs[1:]: v['status']=Status.STANDBY; v['done']=True
        last=3
        for _ in range(3):
            self.e.step([0]*6)
            distance=max(abs(u['position'][k]-10) for k in (0,1))
            self.assertLess(distance,last); last=distance
        np.testing.assert_array_equal(before,self.e.uncertainty_matrix)
        self.assertIn(u['status'],(Status.CHARGING,Status.WAITING))

    def test_charging_monotonic_and_full(self):
        u=self.move_to_vehicle_zone(); u['position']=(10,10)
        self.e.step([0]*6); old=u['energy']
        self.assertEqual(u['status'],Status.CHARGING)
        for _ in range(50):
            self.e.step([0]*6)
            self.assertGreaterEqual(u['energy'],old)
            self.assertLessEqual(u['energy'],self.e.initial_energy)
            old=u['energy']
            if u['status']==Status.STANDBY: break
        self.assertEqual(u['status'],Status.STANDBY)

    def test_fifo_capacity_one(self):
        for i in (0,1,2):
            u=self.e.uavs[i]; u.update(status=Status.RETURNING,position=(10,10),energy=self.e.initial_energy-100,done=True)
        self.e.step([0]*6)
        self.assertEqual([u['status'] for u in self.e.uavs[:3]], [Status.CHARGING,Status.WAITING,Status.WAITING])
        self.e.step([0]*6)
        self.assertEqual(self.e.uavs[1]['status'],Status.CHARGING)
        self.assertEqual(self.e.uavs[2]['status'],Status.WAITING)

    def test_inactive_graph_edges_and_mapping(self):
        g=self.e.get_obs_2()
        self.assertEqual(g.slot_id[:6].tolist(), list(range(6)))
        self.assertTrue((g.x[4:6]==0).all())
        self.assertFalse(((g.edge_index==4)|(g.edge_index==5)).any())
        self.e.eligible={0,2}
        actions=[0,4,0,4,4,4]
        old=[u['energy'] for u in self.e.uavs]
        self.e.step_offload(actions)
        self.assertLess(self.e.uavs[0]['energy'], old[0]); self.assertLess(self.e.uavs[2]['energy'], old[2])
        for i in (1,3,4,5): self.assertEqual(self.e.uavs[i]['energy'],old[i])

    def test_parking_does_not_mark_vehicle_searched(self):
        self.assertEqual(self.e.uncertainty_matrix[10,10], 1)

    def test_returning_does_not_offload(self):
        u=self.move_to_vehicle_zone()
        self.e.step([0]*6)
        before=u['energy']
        _,reward,_=self.e.step_offload([4]*6)
        self.assertEqual(u['energy'], before)
        self.assertEqual(reward[0], 0)

    def test_exhaustion_does_not_teleport_or_go_negative(self):
        u=self.move_to_vehicle_zone(); u['energy']=1
        self.e.step([0]*6)
        self.assertEqual(u['position'], (7,10))
        self.assertGreaterEqual(u['energy'],0)
        self.assertTrue(any(e['event_type']=='energy_exhausted_before_return' for e in self.e.events))

    def test_slot_action_mapping(self):
        self.e.eligible={0,1,2,3}
        self.e.get_obs_2()
        self.e.step_offload([0,1,2,3,4,4])
        self.assertEqual([u['offload'] for u in self.e.uavs[:4]], [0,1,2,3])
        self.assertEqual([u['offload'] for u in self.e.uavs[4:]], [0,0])

    def test_no_premature_termination(self):
        for u in self.e.uavs: u['status']=Status.STANDBY; u['done']=True
        _,_,done=self.e.step([0]*6)
        self.assertFalse(done)

    def test_original_environment_still_works(self):
        env=UAVEnv(N=4); s=env.reset(seed=42)
        self.assertEqual(s.shape,(4,3,15,15))
        self.assertEqual(len(env.step([0]*4)[1]),4)

if __name__ == '__main__': unittest.main()
