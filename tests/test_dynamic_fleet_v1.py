import os
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT/'Our_experiment/HCSAC')
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleetV1 import DynamicFleetV1Config, DynamicFleetV1Env
from Our_experiment.HCSAC.ENV.UAVenv_DynamicFleet import Status, DynamicFleetConfig, DynamicFleetEnv
from Our_experiment.HCSAC.ENV.fleet_scheduler import SchedulerMode, FleetSlot, ReturnRequest, TargetAwareScheduler
from Our_experiment.HCSAC.dynamic_fleet_v1_experiment import fleet_deficit, fleet_surplus, sample_seeds
from Our_experiment.HCSAC.fleet_metrics import request_accounting

class V1Tests(unittest.TestCase):
    def make(self, mode=SchedulerMode.TARGET_AWARE, **kw):
        e=DynamicFleetV1Env(DynamicFleetV1Config(scheduler_mode=mode,**kw))
        positions=[(9,9),(9,11),(11,9),(11,11),(10,10),(10,10)]
        e.reset(seed=42,positions=positions)
        return e

    def test_scheduler_modes_are_distinct(self):
        out=[]
        for mode in SchedulerMode:
            e=self.make(mode); e.uavs[0]['status']=Status.RETURNING; e.uavs[0]['done']=True
            e.uavs[0]['position']=(8,9); e.step([0]*6)
            out.append(sum(x['event_type']=='reserve_launched' for x in e.events))
        self.assertEqual(out,[0,0,1])

    def test_no_replacement_never_launches_reserve(self):
        e=self.make(SchedulerMode.NO_REPLACEMENT); e.uavs[0]['energy']=0.2*e.initial_energy
        for _ in range(20): e.step([0]*6); e.step_offload([0]*6)
        self.assertFalse(any(x['event_type']=='reserve_launched' for x in e.events))

    def test_naive_threshold_preserves_v0_behavior(self):
        e=self.make(SchedulerMode.NAIVE_THRESHOLD); e.uavs[0]['energy']=0.44*e.initial_energy
        e.step([0]*6); e.step([0]*6)
        self.assertEqual(sum(x['event_type']=='reserve_launched' for x in e.events),1)
        self.assertTrue(e.uavs[0]['replacement_requested'])

    def test_target_aware_launches_only_real_deficit(self):
        e=self.make(); e.uavs[0]['energy']=0.44*e.initial_energy; e.step([0]*6)
        self.assertEqual(e.get_fleet_counts()['search_active_count'],5)
        for _ in range(2): e.step([0]*6)
        self.assertEqual(sum(x['event_type']=='reserve_launched' for x in e.events),1)

    def test_existing_replacement_prevents_duplicate_launch(self):
        scheduler=TargetAwareScheduler()
        slots=tuple([FleetSlot(i,'active',1) for i in range(5)]+[FleetSlot(5,'standby',1)])
        d=scheduler.decide(slots,(ReturnRequest(0,1,4),),step=2,target=4,minimum_energy_ratio=.95,handover_min_steps=2)
        self.assertEqual(d.projected_search_count,4); self.assertEqual(d.launch_uav_ids,[])

    def test_charged_standby_fills_existing_target_deficit(self):
        e=self.make(); u=e.uavs[0]; u.update(status=Status.CHARGING,done=True,position=(10,10),energy=e.initial_energy-1)
        for v in e.uavs[4:]: v['energy']=0
        e.step([0]*6); self.assertEqual(u['status'],Status.STANDBY)
        e.step([0]*6); self.assertEqual(u['status'],Status.ACTIVE)
        self.assertEqual(u['activation_reason'],'post_charge_replenishment')

    def test_target_aware_does_not_launch_without_deficit(self):
        e=self.make(); e.step([0]*6)
        self.assertEqual(e.get_fleet_counts()['standby_count'],2)

    def test_two_pending_returns_launch_at_most_two_replacements(self):
        e=self.make(); e.uavs[0]['energy']=.44*e.initial_energy; e.uavs[1]['energy']=.44*e.initial_energy
        e.step([0]*6)
        self.assertEqual(sum(x['event_type']=='reserve_launched' for x in e.events),2)

    def test_replacement_binding_is_one_to_one(self):
        e=self.make(); e.uavs[0]['energy']=.44*e.initial_energy; e.uavs[1]['energy']=.44*e.initial_energy
        e.step([0]*6)
        links=[e.uavs[i]['replacement_uav_id'] for i in (0,1)]
        self.assertEqual(len(set(links)),2); self.assertEqual({e.uavs[j]['replacement_for'] for j in links},{0,1})

    def test_physical_charging_increment(self):
        e=self.make(charging_power_w=600,charging_efficiency=.9)
        u=e.uavs[0]; u.update(status=Status.CHARGING,done=True,position=(10,10),energy=100000)
        e.step([0]*6)
        self.assertAlmostEqual(u['energy'],100000+600*.9*e.T)
        self.assertAlmostEqual(e.energy_stats['total_grid_energy_input_j'],600*e.T)

    def test_charging_energy_never_exceeds_capacity(self):
        e=self.make(); u=e.uavs[0]; u.update(status=Status.CHARGING,done=True,position=(10,10),energy=e.initial_energy-1)
        e.step([0]*6); self.assertEqual(u['energy'],e.initial_energy)
        self.assertAlmostEqual(e.energy_stats['total_grid_energy_input_j'],1/.9)

    def test_minimum_launch_energy_ratio(self):
        e=self.make(); e.uavs[0]['status']=Status.RETURNING; e.uavs[0]['done']=True; e.uavs[0]['position']=(8,9)
        e.uavs[4]['energy']=.94*e.initial_energy
        e.uavs[5]['energy']=.96*e.initial_energy
        e.step([0]*6)
        self.assertEqual(e.uavs[4]['status'],Status.STANDBY); self.assertEqual(e.uavs[5]['status'],Status.ACTIVE)

    def test_integrated_fleet_deficit_metric(self):
        self.assertEqual(fleet_deficit([4,3,2,6],4),3)

    def test_integrated_fleet_surplus_metric(self):
        self.assertEqual(fleet_surplus([4,3,5,6],4),3)

    def test_shared_seed_configuration(self):
        self.assertEqual(sample_seeds(4800,42),{'wind_seed':4800,'terrain_seed':42,'infra_seed':42,'trajectory_seed':42})
        a=self.make(); a.reset(seed=42,wind_seed=4800,terrain_seed=42,infra_seed=42)
        b=self.make(); b.reset(seed=42,wind_seed=4800,terrain_seed=42,infra_seed=42)
        np.testing.assert_array_equal(a.task_matrix,b.task_matrix); np.testing.assert_array_equal(a.wind_u,b.wind_u)
        np.testing.assert_array_equal(a.gbs_position,b.gbs_position)

    def test_inactive_uav_offload_count_is_zero(self):
        e=self.make(); e.step([0]*6); before=[u['energy'] for u in e.uavs]
        _,r,_=e.step_offload([4]*6)
        for i in set(range(6))-e.eligible:
            self.assertEqual(before[i],e.uavs[i]['energy']); self.assertEqual(r[i],0)

    def test_naive_matches_v0_rollout_when_charging_increment_matches(self):
        positions=[(9,9),(9,11),(11,9),(11,11),(10,10),(10,10)]
        old=DynamicFleetEnv(DynamicFleetConfig())
        new=DynamicFleetV1Env(DynamicFleetV1Config(scheduler_mode=SchedulerMode.NAIVE_THRESHOLD))
        new.config.charging_power_w=old.config.charge_rate_per_step/(new.config.charging_efficiency*new.T)
        old.reset(seed=42,positions=positions); new.reset(seed=42,positions=positions)
        old.uavs[0]['energy']=new.uavs[0]['energy']=old.initial_energy*.44
        for _ in range(20):
            old.step([0]*6); new.step([0]*6)
            old.step_offload([0]*6); new.step_offload([0]*6)
            self.assertEqual(old.get_uav_statuses(),new.get_uav_statuses())
            self.assertEqual([u['position'] for u in old.uavs],[u['position'] for u in new.uavs])
            np.testing.assert_allclose([u['energy'] for u in old.uavs],[u['energy'] for u in new.uavs],rtol=0,atol=1e-7)
            np.testing.assert_array_equal(old.uncertainty_matrix,new.uncertainty_matrix)

    def test_standby_selection_energy_then_wait_then_id(self):
        scheduler=TargetAwareScheduler()
        slots=(FleetSlot(0,'active',1),FleetSlot(1,'active',1),FleetSlot(2,'active',1),
               FleetSlot(3,'standby',.97,0),FleetSlot(4,'standby',.99,5),FleetSlot(5,'standby',.99,2))
        d=scheduler.decide(slots,(),step=10,target=4,minimum_energy_ratio=.95,handover_min_steps=2)
        self.assertEqual(d.launch_uav_ids,[5]); self.assertEqual(d.launch_selection_rank[5],1)
        self.assertEqual(d.decision_reason[5],'restore_target_count')

    def test_safety_return_overrides_handover(self):
        scheduler=TargetAwareScheduler()
        slots=(FleetSlot(0,'active',.2,safety_return=True,replacement_launch_step=5),
               FleetSlot(1,'active',1),FleetSlot(2,'active',1),FleetSlot(3,'active',1),FleetSlot(4,'active',1))
        d=scheduler.decide(slots,(ReturnRequest(0,5,4),),step=5,target=4,minimum_energy_ratio=.95,handover_min_steps=2)
        self.assertEqual(d.return_uav_ids,[0]); self.assertEqual(d.launch_uav_ids,[])

    def test_target_aware_no_standby_waits_handover_unless_unsafe(self):
        scheduler=TargetAwareScheduler()
        slots=tuple(FleetSlot(i,'active',.2 if i==0 else 1,threshold_return=(i==0)) for i in range(4))
        request=(ReturnRequest(0,5),)
        d=scheduler.decide(slots,request,step=5,target=4,minimum_energy_ratio=.95,handover_min_steps=2)
        self.assertEqual(d.return_uav_ids,[])
        d=scheduler.decide(slots,request,step=7,target=4,minimum_energy_ratio=.95,handover_min_steps=2)
        self.assertEqual(d.return_uav_ids,[0])

    def test_vectorized_observations_equal_v0_at_edges_and_occupancy(self):
        e=self.make()
        for positions in [[(0,0),(19,19),(0,19),(19,0)],[(10,10),(10,11),(11,10),(12,12)]]:
            for u,pos in zip(e.uavs,positions): u['position']=pos
            for i in range(4): e.uavs[i]['status']=Status.ACTIVE; e.uavs[i]['done']=False
            e.uavs[2]['status']=Status.RETURNING; e.uavs[2]['done']=True
            np.testing.assert_array_equal(e._get_obs(),DynamicFleetEnv._get_obs(e))

    def test_remote_and_local_fallback_energy_attribution(self):
        remote=self.make(); remote.step([0]*6)
        remote.gbs_position=np.array([10.,10.]); remote.get_obs_2()
        remote.step_offload([1]*6)
        self.assertGreater(remote.energy_stats['communication_energy_j'],0)
        self.assertEqual(remote.energy_stats['local_compute_energy_j'],0)
        self.assertEqual(remote.effective_offload_counts[1],len(remote.eligible))
        fallback=self.make(); fallback.step([0]*6)
        fallback.gbs_position=np.array([1000.,1000.]); fallback.get_obs_2()
        fallback.step_offload([1]*6)
        self.assertEqual(fallback.energy_stats['communication_energy_j'],0)
        self.assertGreater(fallback.energy_stats['local_compute_energy_j'],0)
        self.assertEqual(fallback.effective_offload_counts[0],len(fallback.eligible))

    def test_request_accounting_distinguishes_unserved_and_assigned(self):
        def ev(kind,uid,**metadata):
            return {'event_type':kind,'uav_id':uid,'metadata':metadata}
        events=[ev('return_request_created',0),ev('active_to_returning',0),
                ev('return_request_created',1),ev('replacement_assigned',1,replacement_uav_id=4),
                ev('return_request_created',2)]
        self.assertEqual(request_accounting(events),{'unresolved_return_requests':2,
            'returned_without_replacement':1,'unassigned_pending_return_requests':1,'active_pending_return_requests':2})

    def test_effective_offload_fallback_accounting(self):
        e=self.make(); e.step([0]*6); e.get_obs_2(); e.step_offload([3]*6)
        self.assertEqual(sum(e.effective_offload_counts),len(e.eligible))
        self.assertAlmostEqual(e.energy_stats['local_compute_energy_j']+e.energy_stats['communication_energy_j'],e.energy_stats['offload_energy_j'])
        self.assertGreater(e.effective_offload_counts[0],0)

if __name__=='__main__': unittest.main()
