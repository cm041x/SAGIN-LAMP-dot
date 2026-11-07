"""Pure fleet scheduling decisions: no environment or UAV mutation."""
from dataclasses import dataclass, field
from enum import Enum

class SchedulerMode(str, Enum):
    NO_REPLACEMENT = 'no_replacement'
    NAIVE_THRESHOLD = 'naive_threshold'
    TARGET_AWARE = 'target_aware'

@dataclass(frozen=True)
class ReturnRequest:
    uav_id: int
    request_step: int
    replacement_uav_id: int | None = None
    reason: str = 'low_energy'

@dataclass(frozen=True)
class FleetSlot:
    uav_id: int
    status: str
    energy_ratio: float
    standby_since_step: int = 0
    full_charge_step: int | None = None
    safety_return: bool = False
    threshold_return: bool = False
    replacement_launch_step: int | None = None

@dataclass
class FleetDecision:
    launch_uav_ids: list[int] = field(default_factory=list)
    return_uav_ids: list[int] = field(default_factory=list)
    request_to_replacement_mapping: dict[int, int] = field(default_factory=dict)
    projected_search_count: int = 0
    launch_deficit: int = 0
    decision_reason: dict[int, str] = field(default_factory=dict)
    launch_selection_rank: dict[int, int] = field(default_factory=dict)
    unavailable_request_ids: list[int] = field(default_factory=list)

class FleetScheduler:
    def decide(self, slots: tuple[FleetSlot, ...], requests: tuple[ReturnRequest, ...], *, step: int,
               target: int, minimum_energy_ratio: float, handover_min_steps: int) -> FleetDecision:
        raise NotImplementedError

    @staticmethod
    def _returns(slots: tuple[FleetSlot, ...], requests: tuple[ReturnRequest, ...], step: int,
                 handover_min_steps: int, require_request_age: bool = False) -> list[int]:
        by_id = {r.uav_id: r for r in requests}
        result=[]
        for s in slots:
            if s.status != 'active': continue
            r=by_id.get(s.uav_id)
            if r is None:
                handover_done = True
            elif r.replacement_uav_id is None:
                handover_done = not require_request_age or step-r.request_step >= handover_min_steps
            else:
                handover_done = s.replacement_launch_step is not None and step-s.replacement_launch_step >= handover_min_steps
            if s.safety_return or (s.threshold_return and handover_done): result.append(s.uav_id)
        return result

class NoReplacementScheduler(FleetScheduler):
    def decide(self, slots: tuple[FleetSlot, ...], requests: tuple[ReturnRequest, ...], *, step: int,
               target: int, minimum_energy_ratio: float, handover_min_steps: int) -> FleetDecision:
        active=sum(s.status=='active' for s in slots)
        return FleetDecision(return_uav_ids=self._returns(slots, requests, step, handover_min_steps),
                             projected_search_count=active, launch_deficit=max(0,target-active))

class NaiveThresholdScheduler(FleetScheduler):
    def decide(self, slots: tuple[FleetSlot, ...], requests: tuple[ReturnRequest, ...], *, step: int,
               target: int, minimum_energy_ratio: float, handover_min_steps: int) -> FleetDecision:
        active=sum(s.status=='active' for s in slots)
        d=FleetDecision(projected_search_count=active, launch_deficit=max(0,target-active))
        available=[s for s in slots if s.status=='standby' and s.energy_ratio >= minimum_energy_ratio]
        for req in sorted(requests,key=lambda r:r.uav_id):
            if req.replacement_uav_id is not None: continue
            if not available:
                d.unavailable_request_ids.append(req.uav_id); continue
            slot=available.pop(0)
            d.launch_uav_ids.append(slot.uav_id)
            d.request_to_replacement_mapping[req.uav_id]=slot.uav_id
            d.decision_reason[slot.uav_id]='replace_pending_return'
            d.launch_selection_rank[slot.uav_id]=len(d.launch_uav_ids)
        # Newly allocated replacements need handover unless safety forces return.
        augmented=tuple(ReturnRequest(r.uav_id,r.request_step,d.request_to_replacement_mapping.get(r.uav_id,r.replacement_uav_id),r.reason) for r in requests)
        marked=tuple(FleetSlot(**{**s.__dict__,'replacement_launch_step':step}) if s.uav_id in d.request_to_replacement_mapping else s for s in slots)
        d.return_uav_ids=self._returns(marked,augmented,step,handover_min_steps)
        return d

class TargetAwareScheduler(FleetScheduler):
    def decide(self, slots: tuple[FleetSlot, ...], requests: tuple[ReturnRequest, ...], *, step: int,
               target: int, minimum_energy_ratio: float, handover_min_steps: int) -> FleetDecision:
        active={s.uav_id for s in slots if s.status=='active'}
        pending=[r for r in requests if r.uav_id in active]
        projected=len(active)-len(pending)
        deficit=max(0,target-projected)
        d=FleetDecision(projected_search_count=projected,launch_deficit=deficit)
        available=sorted((s for s in slots if s.status=='standby' and s.energy_ratio>=minimum_energy_ratio),
                         key=lambda s:(-s.energy_ratio,s.standby_since_step,s.uav_id))
        unassigned=sorted((r for r in pending if r.replacement_uav_id is None),key=lambda r:(r.request_step,r.uav_id))
        for rank,s in enumerate(available[:deficit],1):
            d.launch_uav_ids.append(s.uav_id); d.launch_selection_rank[s.uav_id]=rank
            if unassigned:
                req=unassigned.pop(0)
                d.request_to_replacement_mapping[req.uav_id]=s.uav_id
                d.decision_reason[s.uav_id]='replace_pending_return'
            else:
                d.decision_reason[s.uav_id]='post_charge_replenishment' if s.full_charge_step is not None else 'restore_target_count'
        d.unavailable_request_ids=[r.uav_id for r in unassigned]
        augmented=tuple(ReturnRequest(r.uav_id,r.request_step,d.request_to_replacement_mapping.get(r.uav_id,r.replacement_uav_id),r.reason) for r in requests)
        marked=tuple(FleetSlot(**{**s.__dict__,'replacement_launch_step':step}) if s.uav_id in d.request_to_replacement_mapping else s for s in slots)
        d.return_uav_ids=self._returns(marked,augmented,step,handover_min_steps,require_request_age=True)
        return d

def make_scheduler(mode: SchedulerMode) -> FleetScheduler:
    return {SchedulerMode.NO_REPLACEMENT:NoReplacementScheduler,
            SchedulerMode.NAIVE_THRESHOLD:NaiveThresholdScheduler,
            SchedulerMode.TARGET_AWARE:TargetAwareScheduler}[SchedulerMode(mode)]()
