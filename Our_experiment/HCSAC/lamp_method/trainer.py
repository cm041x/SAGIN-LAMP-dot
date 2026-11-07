"""Recurrent team SAC with raw-prefix reconstruction and frozen collection episodes."""
import copy
import math
import random
import numpy as np
import torch
from torch import nn
from .policy import Actor, make_encoder, encode
from .critics import JointCriticHead, event_critic_loss


class TeamCritic(nn.Module):
    def __init__(self, variant: str, hidden_dim: int = 128):
        super().__init__()
        self.encoder=make_encoder(variant,hidden_dim)
        self.head=JointCriticHead(hidden_dim,event_conditioned=variant=='lamp_lifecycle_ecc')


def reconstruct(encoder, prefix, unroll, *, grad: bool, add_last: bool):
    """Recompute history with current network; only unroll carries gradients."""
    hidden=None
    with torch.no_grad():
        for t in prefix: _,hidden=encode(encoder,t['snapshot'],hidden)
    if hidden is not None: hidden=hidden.detach()
    snapshots=[t['snapshot'] for t in unroll]
    if add_last and unroll[-1]['next_snapshot'] is not None:
        snapshots.append(unroll[-1]['next_snapshot'])
    result=[]
    with torch.set_grad_enabled(grad):
        for snapshot in snapshots:
            h,hidden=encode(encoder,snapshot,hidden);result.append(h)
    return result


def sample_actions(probs, mask):
    actions=torch.zeros(6,dtype=torch.long,device=probs.device)
    if mask.any(): actions[mask]=torch.distributions.Categorical(probs[mask]).sample()
    return actions


def entropy_temperature_loss(beta, entropy, ratio=0.5, n_actions=5):
    return beta.exp()*(entropy.detach()-ratio*math.log(n_actions))


class TeamSAC:
    def __init__(self, variant: str, actor_lr=3e-4, critic_lr=3e-4, gamma=.99, tau=.005, alpha=.01,
                 temperature_mode='fixed', entropy_ratio=.5):
        if temperature_mode not in ('fixed','auto'): raise ValueError('invalid temperature mode')
        self.variant=variant;self.gamma=gamma;self.tau=tau;self.alpha=alpha
        self.temperature_mode=temperature_mode;self.entropy_ratio=entropy_ratio
        self.beta=torch.tensor(math.log(alpha),requires_grad=True)
        self.temperature_optimizer=torch.optim.Adam([self.beta],lr=1e-4)
        # Independent streams keep shared encoder weights aligned across ablations,
        # even when ECC uses a differently sized Q readout.
        base_seed=torch.initial_seed()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((base_seed+101) % (2**63));self.actor=Actor(variant)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((base_seed+202) % (2**63));self.q1=TeamCritic(variant)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((base_seed+303) % (2**63));self.q2=TeamCritic(variant)
        self.tq1=copy.deepcopy(self.q1);self.tq2=copy.deepcopy(self.q2)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=critic_lr)
        self.updates=0

    def state_dict(self):
        return {'variant':self.variant, 'actor':self.actor.state_dict(), 'q1':self.q1.state_dict(),
                'q2':self.q2.state_dict(), 'tq1':self.tq1.state_dict(), 'tq2':self.tq2.state_dict(),
                'actor_optimizer':self.actor_optimizer.state_dict(), 'critic_optimizer':self.critic_optimizer.state_dict(),
                'updates':self.updates, 'temperature_mode':self.temperature_mode,
                'beta':self.beta.detach().clone(), 'temperature_optimizer':self.temperature_optimizer.state_dict()}

    def load_state_dict(self, state):
        if state['variant']!=self.variant: raise ValueError('checkpoint variant mismatch')
        if state['temperature_mode']!=self.temperature_mode: raise ValueError('temperature mode mismatch')
        for key in ('actor','q1','q2','tq1','tq2'): getattr(self,key).load_state_dict(state[key])
        self.actor_optimizer.load_state_dict(state['actor_optimizer']);self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.updates=state['updates']
        with torch.no_grad(): self.beta.copy_(state['beta'])
        self.temperature_optimizer.load_state_dict(state['temperature_optimizer'])
        self.alpha=float(self.beta.detach().exp())

    def update(self, sequences: list[tuple[list,list]]) -> dict:
        actor_terms=[];critic_terms=[];labels=[];events=[];hidden_norm=[];entropies=[]
        for sample in sequences:
            if len(sample)==2:
                prefix,seq=sample;sequence_mask=torch.ones(len(seq),dtype=torch.bool)
            else:
                prefix,seq,sequence_mask=sample
            if sequence_mask.shape!=(len(seq),) or sequence_mask.dtype!=torch.bool:
                raise ValueError('sequence_mask must be a boolean per timestep')
            valid=int(sequence_mask.sum())
            if not torch.equal(sequence_mask,torch.arange(len(seq))<valid):
                raise ValueError('padding must be a suffix, not interleaved')
            seq=seq[:valid]  # Padded steps never enter hidden reconstruction or loss.
            if not seq: continue
            ah=reconstruct(self.actor.encoder,prefix,seq,grad=True,add_last=True)
            q1h=reconstruct(self.q1.encoder,prefix,seq,grad=True,add_last=False)
            q2h=reconstruct(self.q2.encoder,prefix,seq,grad=True,add_last=False)
            tq1h=reconstruct(self.tq1.encoder,prefix,seq,grad=False,add_last=True)
            tq2h=reconstruct(self.tq2.encoder,prefix,seq,grad=False,add_last=True)
            for t,tr in enumerate(seq):
                snapshot=tr['snapshot']; mask=snapshot.action_mask; actions=tr['actions']
                target=torch.tensor(float(tr['reward']))
                if not tr['terminal']:
                    with torch.no_grad():
                        next_mask=tr['next_snapshot'].action_mask
                        next_probs=self.actor.probabilities(ah[t+1].detach())
                        next_actions=sample_actions(next_probs,next_mask)
                        v1=self.tq1.head(tq1h[t+1],next_actions,next_mask)[0]
                        v2=self.tq2.head(tq2h[t+1],next_actions,next_mask)[0]
                        logp=torch.log(next_probs.clamp_min(1e-8))
                        entropy_term=logp[next_mask].gather(-1,next_actions[next_mask,None]).sum() if next_mask.any() else torch.tensor(0.)
                        target=target+self.gamma*(torch.minimum(v1,v2)-self.alpha*entropy_term)
                event=torch.tensor([tr['event']],dtype=torch.long)
                for q,h in ((self.q1,q1h[t]),(self.q2,q2h[t])):
                    value,logits,values=q.head(h,actions,mask)
                    if logits is None: critic_terms.append((value-target.detach()).square())
                    else:
                        critic_terms.append(event_critic_loss(logits.unsqueeze(0),values.unsqueeze(0),event,target.reshape(1)))
                        labels.append(tr['event']);events.append(torch.softmax(logits.detach(),-1))
                hidden_norm.append(ah[t].detach().norm().item())
                if mask.any():
                    probs=self.actor.probabilities(ah[t]);other=sample_actions(probs.detach(),mask)
                    indices=torch.where(mask)[0]; count=len(indices)*5
                    alternatives=other.repeat(count,1)
                    for k,i in enumerate(indices.tolist()): alternatives[k*5:(k+1)*5,i]=torch.arange(5)
                    masks=mask.expand(count,-1)
                    with torch.no_grad():
                        v1=self.q1.head(q1h[t].detach().expand(count,-1,-1),alternatives,masks)[0]
                        v2=self.q2.head(q2h[t].detach().expand(count,-1,-1),alternatives,masks)[0]
                        values=torch.minimum(v1,v2).reshape(len(indices),5)
                    p=probs[indices]; logp=torch.log(p.clamp_min(1e-8))
                    actor_terms.append((p*(self.alpha*logp-values)).sum()/6)
                    entropies.append(-(p.detach()*logp.detach()).sum(-1).mean())
        if not critic_terms: raise ValueError('Empty sequence batch')
        critic_loss=torch.stack(critic_terms).mean()
        self.critic_optimizer.zero_grad(set_to_none=True);critic_loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for net in (self.q1,self.q2) for p in net.parameters()):
            raise FloatingPointError('Non-finite critic gradient')
        self.critic_optimizer.step()
        actor_loss=None; conditioner_gradient=None
        if actor_terms:
            actor_loss=torch.stack(actor_terms).mean();self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.actor.parameters()):
                raise FloatingPointError('Non-finite actor gradient')
            conditioner=getattr(self.actor.encoder,'conditioner',None)
            if conditioner is not None:
                conditioner_gradient=float(sum(p.grad.norm().item() for p in conditioner.parameters() if p.grad is not None))
            self.actor_optimizer.step()
        if self.temperature_mode=='auto' and entropies:
            temperature_loss=entropy_temperature_loss(self.beta,torch.stack(entropies).mean(),self.entropy_ratio)
            self.temperature_optimizer.zero_grad(set_to_none=True);temperature_loss.backward();self.temperature_optimizer.step()
            self.alpha=float(self.beta.detach().exp())
        with torch.no_grad():
            for target,source in ((self.tq1,self.q1),(self.tq2,self.q2)):
                for t,s in zip(target.parameters(),source.parameters()): t.lerp_(s,self.tau)
        self.updates+=1
        brier=None;ce=None;event_accuracy=None
        if events:
            p=torch.stack(events); lab=torch.tensor(labels); truth=torch.nn.functional.one_hot(lab,8)
            brier=(p-truth).square().sum(-1).mean().item()
            ce=-p[torch.arange(len(lab)),lab].clamp_min(1e-8).log().mean().item()
            event_accuracy=(p.argmax(-1)==lab).float().mean().item()
        return {'critic_loss':critic_loss.item(),'actor_loss':actor_loss.item() if actor_loss is not None else None,
                'hidden_norm':float(np.mean(hidden_norm)),'event_brier':brier,'event_ce':ce,
                'event_counts':[labels.count(i) for i in range(8)],'updates':self.updates,
                'event_accuracy':event_accuracy,'conditioner_gradient_norm':conditioner_gradient,'alpha':self.alpha,
                'entropy':torch.stack(entropies).mean().item() if entropies else None}
