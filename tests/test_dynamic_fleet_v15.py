import os
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT/'Our_experiment/HCSAC')
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import Status
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleetV1 import DynamicFleetV1Config,DynamicFleetV1Env
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleetV15 import DynamicFleetV15Config,DynamicFleetV15Env
from Our_experiment.HCSAC.ENV.fleet_scheduler import SchedulerMode
from Our_experiment.HCSAC.ENV.handover_policy import HandoverMode,HandoverPolicyConfig,responsibility_partition,choose_target,waypoint_action,distance
from Our_experiment.HCSAC.dynamic_fleet_v15_metrics import runs_reaching_one,coverage_gain_per_active_step,pairwise_visit_overlap,proximity_pairs,contention_index

class V15Tests(unittest.TestCase):
    def make(self, handover=HandoverMode.NONE, **kwargs):
        e=DynamicFleetV15Env(DynamicFleetV15Config(handover_mode=handover,**kwargs))
        e.reset(seed=42,positions=[(9,9),(9,11),(11,9),(11,11),(10,10),(10,10)])
        return e
    def start_handover(self, **kw):
        e=self.make(HandoverMode.UNCERTAINTY_WAYPOINT,**kw)
        e.uavs[0]['energy']=.44*e.initial_energy
        e.step([0]*6)
        ids=[i for i,u in enumerate(e.uavs) if u['handover_active']]
        self.assertEqual(len(ids),1)
        return e,ids[0]
    def test_capacity_two_allows_two_simultaneous_chargers(self):
        e=self.make(charging_capacity=2)
        for u in e.uavs[:3]: u.update(status=Status.RETURNING,done=True,position=(10,10),energy=100000)
        e.step([0]*6); self.assertEqual(e.get_fleet_counts()['charging_count'],2)
    def test_capacity_one_preserves_fifo_behavior(self):
        e=self.make(charging_capacity=1)
        for u in e.uavs[:3]: u.update(status=Status.RETURNING,done=True,position=(10,10),energy=e.initial_energy-1)
        e.step([0]*6); e.step([0]*6); self.assertEqual(e.uavs[1]['status'],Status.CHARGING); self.assertEqual(e.uavs[2]['status'],Status.WAITING)
    def test_target_three_scheduler_uses_configured_target(self):
        e=self.make(target_search_uavs=3);e.step([0]*6);self.assertEqual(e.get_fleet_counts()['search_active_count'],4)
    def test_target_four_scheduler_uses_configured_target(self):
        e=self.make(target_search_uavs=4);e.uavs[0].update(status=Status.RETURNING,done=True,position=(8,9));e.step([0]*6);self.assertEqual(e.get_fleet_counts()['search_active_count'],4)
    def test_target_parameter_is_not_hardcoded(self):
        e=self.make(target_search_uavs=5);e.step([0]*6);self.assertEqual(e.get_fleet_counts()['search_active_count'],5)
    def test_capacity_parameter_is_not_hardcoded(self):
        e=self.make(charging_capacity=2)
        for u in e.uavs[:3]:u.update(status=Status.RETURNING,done=True,position=(10,10),energy=10000)
        e.step([0]*6);self.assertEqual(e.get_fleet_counts()['waiting_charge_count'],1)
    def test_runs_reaching_one_uav_metric(self):self.assertEqual(runs_reaching_one([4,2,1,2]),1);self.assertEqual(runs_reaching_one([4,2,3]),0)
    def test_coverage_gain_per_active_step_metric(self):self.assertEqual(coverage_gain_per_active_step(20,40),.5);self.assertEqual(coverage_gain_per_active_step(0,0),0)
    def test_handover_modes_are_distinct(self):
        e,i=self.start_handover();self.assertTrue(e.uavs[i]['handover_active'])
        n=self.make();n.uavs[0]['energy']=.44*n.initial_energy;n.step([0]*6);self.assertFalse(any(x['event_type']=='handover_started' for x in n.events))
    def test_no_handover_preserves_v1_behavior(self):
        a=self.make(); b=DynamicFleetV1Env(DynamicFleetV1Config()); b.reset(seed=42,positions=[(9,9),(9,11),(11,9),(11,11),(10,10),(10,10)])
        for e in (a,b):e.uavs[0]['energy']=.44*e.initial_energy
        for _ in range(15):
            a.step([0]*6);b.step([0]*6);a.get_obs_2();b.get_obs_2();a.step_offload([0]*6);b.step_offload([0]*6)
            self.assertEqual(a.get_uav_statuses(),b.get_uav_statuses());np.testing.assert_array_equal(a.uncertainty_matrix,b.uncertainty_matrix)
            np.testing.assert_allclose([u['energy'] for u in a.uavs],[u['energy'] for u in b.uavs])
    def test_responsibility_partition_assigns_each_cell_once(self):
        a=responsibility_partition({0:(0,0),1:(19,19)});self.assertEqual(a.shape,(20,20));self.assertTrue(set(a.ravel())=={0,1})
    def test_responsibility_partition_tie_break_is_deterministic(self):self.assertEqual(responsibility_partition({4:(0,0),1:(0,0)})[3,3],1)
    def test_handover_target_is_inside_outgoing_region_when_available(self):
        p={0:(0,0),1:(19,19)};t,f,_=choose_target(0,(10,10),p,np.ones((20,20)),(10,10),HandoverPolicyConfig());self.assertFalse(f);self.assertEqual(responsibility_partition(p)[t],0)
    def test_global_fallback_when_region_has_no_candidate(self):
        p={0:(0,0),1:(19,19)};u=np.zeros((20,20));u[18,18]=1;t,f,_=choose_target(0,(10,10),p,u,(10,10),HandoverPolicyConfig());self.assertTrue(f);self.assertEqual(t,(18,18))
    def test_replacement_has_at_most_one_waypoint(self):
        e,i=self.start_handover()
        with self.assertRaises(RuntimeError):e._create_waypoint(0,i)
    def test_outgoing_uav_has_at_most_one_replacement(self):
        e,i=self.start_handover(); self.assertEqual(e.uavs[0]['replacement_uav_id'],i); self.assertEqual(sum(u['replacement_for']==0 for u in e.uavs),1)
    def test_waypoint_move_reduces_target_distance(self):
        e=self.make(); i=4;e.uavs[i].update(status=Status.ACTIVE,done=False,position=(10,10)); [u.update(position=p) for u,p in zip(e.uavs[:4],[(0,0),(0,19),(19,0),(19,19)])]; e.wind_u[:]=0; e.wind_v[:]=0; a=waypoint_action(e,i,(10,15),e.policy); d=e.directions[a];self.assertLess(distance((10+d[0],10+d[1]),(10,15)),5)
    def test_waypoint_never_teleports(self):
        e,i=self.start_handover();old=e.uavs[i]['position'];e.step([0]*6);p=e.uavs[i]['position'];self.assertLessEqual(distance(old,p),1)
    def test_waypoint_releases_on_target_reached(self):
        e,i=self.start_handover();e.uavs[i]['handover_target_cell']=(10,11);e.uavs[i]['handover_initial_target_uncertainty']=1;e.uncertainty_matrix[10,11]=1
        e.uavs[1]['position']=(1,1);e.step([0]*6)
        self.assertIn(e.uavs[i]['handover_end_reason'],('target_reached','target_already_resolved'))
    def test_waypoint_releases_on_timeout(self):
        e,i=self.start_handover(max_handover_steps=1);e.uavs[i]['handover_target_cell']=(0,0);e.step([0]*6);self.assertEqual(e.uavs[i]['handover_end_reason'],'timeout')
    def test_waypoint_releases_when_replacement_returns(self):
        e,i=self.start_handover();e.uavs[i]['energy']=1;e.step([0]*6);self.assertEqual(e.uavs[i]['handover_end_reason'],'replacement_returning')
    def test_handover_uav_still_offloads_when_active(self):
        e,i=self.start_handover();e.step([0]*6);e.get_obs_2();n=sum(e.offload_target_counts);e.step_offload([0]*6);self.assertIn(i,e.eligible);self.assertGreater(sum(e.offload_target_counts),n)
    def test_inactive_uav_never_offloads(self):
        e=self.make();e.step([0]*6);before=e.uavs[4]['energy'];e.get_obs_2();_,r,_=e.step_offload([0]*6);self.assertEqual(r[4],0);self.assertEqual(e.uavs[4]['energy'],before)
    def test_pairwise_overlap_metric(self):
        a=np.zeros((2,2,2));a[0,0,:]=1;a[1,0,1]=1;self.assertEqual(pairwise_visit_overlap(a),(.5,.5))
    def test_proximity_conflict_metric(self):self.assertEqual(proximity_pairs([(0,0),(1,1),(5,5)]),1)
    def test_target_contention_index(self):self.assertEqual(contention_index([{'active_decisions':4,'requested':[3,1,0,0,0]}]),.5)
    def test_handover_event_schema(self):
        e,i=self.start_handover();ev=next(x for x in e.events if x['event_type']=='handover_target_assigned');self.assertEqual(ev['uav_id'],i);self.assertIn('target',ev['metadata'])
    def test_handover_determinism_under_fixed_seed(self):
        a,i=self.start_handover();b,j=self.start_handover();self.assertEqual(a.uavs[i]['handover_target_cell'],b.uavs[j]['handover_target_cell'])
        for _ in range(3):a.step([0]*6);b.step([0]*6)
        self.assertEqual([u['position'] for u in a.uavs],[u['position'] for u in b.uavs])

if __name__=='__main__':unittest.main()
