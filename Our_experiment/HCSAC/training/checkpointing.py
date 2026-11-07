"""Atomic checkpointing and complete state capture."""
import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch

def module_hash(module) -> str:
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def file_hash(path) -> str:return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def agent_state(agent) -> dict:
    return {**{k:getattr(agent,k).state_dict() for k in ('actor','critic_1','critic_2','target_critic_1','target_critic_2')},
            **{k:getattr(agent,k).state_dict() for k in ('actor_optimizer','critic_1_optimizer','critic_2_optimizer','alpha_optimizer')},
            'log_alpha':agent.log_alpha.detach().cpu(),'skipped_empty_mask_updates':agent.skipped_empty_mask_updates}

def restore_agent(agent,state: dict) -> None:
    for k in ('actor','critic_1','critic_2','target_critic_1','target_critic_2'):getattr(agent,k).load_state_dict(state[k],strict=True)
    for k in ('actor_optimizer','critic_1_optimizer','critic_2_optimizer','alpha_optimizer'):getattr(agent,k).load_state_dict(state[k])
    with torch.no_grad():agent.log_alpha.copy_(state['log_alpha'].to(agent.device))
    agent.skipped_empty_mask_updates=state['skipped_empty_mask_updates']

def rng_state() -> dict:
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state: dict) -> None:
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])

def atomic_save(path: Path, obj) -> None:
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)
