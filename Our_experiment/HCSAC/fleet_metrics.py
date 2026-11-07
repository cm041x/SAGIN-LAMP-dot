"""Derived request accounting from the persisted scheduling event stream."""
from typing import Any

def request_accounting(events: list[dict[str, Any]]) -> dict[str, int]:
    pending: dict[int,bool]={}
    returned_without_replacement=0
    for event in events:
        kind=event['event_type']; owner=event['uav_id']
        if kind=='return_request_created':
            pending[owner]=False
        elif kind=='replacement_assigned' and owner in pending:
            pending[owner]=True
        elif kind=='replacement_binding_released' and event['metadata'].get('reason')=='replacement_returning' and owner in pending:
            pending[owner]=False
        elif kind=='active_to_returning' and owner in pending:
            returned_without_replacement+=int(not pending.pop(owner))
    unassigned_pending=sum(not assigned for assigned in pending.values())
    return {'unresolved_return_requests':returned_without_replacement+unassigned_pending,
            'returned_without_replacement':returned_without_replacement,
            'unassigned_pending_return_requests':unassigned_pending,
            'active_pending_return_requests':len(pending)}
