"""Flight team SAC with raw-prefix reconstruction and frozen collection episodes."""
import copy
import math
import random
import numpy as np
import torch
from torch import nn
from .policy import StableActor, StableEncoder, encode, ACTION_DIM, UAV_COUNT
from Our_experiment.HCSAC.flight_residual.critics import FlightCriticHead, event_critic_loss


class TeamCritic(nn.Module):
    def __init__(self, variant: str, hidden_dim: int = 128, xi: float = 2.):
        super().__init__()
        self.encoder=StableEncoder(variant,hidden_dim,xi)
        self.head=FlightCriticHead(hidden_dim,event_conditioned=False)


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
    actions=torch.zeros(UAV_COUNT,dtype=torch.long,device=probs.device)
    if mask.any(): actions[mask]=torch.distributions.Categorical(probs[mask]).sample()
    return actions


def entropy_temperature_loss(beta, entropy, ratio=0.5, n_actions=ACTION_DIM):
    return beta.exp()*(entropy.detach()-ratio*math.log(n_actions))


class TeamSAC:
    def __init__(self, variant: str, actor_lr=3e-4, critic_lr=3e-4, gamma=.99, tau=.005, alpha=.01,
                 temperature_mode='fixed', entropy_ratio=.5, residual_bound=1., bounded=True, xi=2.):
        if temperature_mode not in ('fixed','auto'): raise ValueError('invalid temperature mode')
        self.variant=variant;self.gamma=gamma;self.tau=tau;self.alpha=alpha
        self.temperature_mode=temperature_mode;self.entropy_ratio=entropy_ratio
        self.beta=torch.tensor(math.log(alpha),requires_grad=True)
        self.temperature_optimizer=torch.optim.Adam([self.beta],lr=1e-4)
        # Independent streams keep shared encoder weights aligned across ablations,
        # even when ECC uses a differently sized Q readout.
        base_seed=torch.initial_seed()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((base_seed+101) % (2**63));self.actor=StableActor(variant,residual_bound=residual_bound,bounded=bounded,xi=xi)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((base_seed+202) % (2**63));self.q1=TeamCritic(variant,xi=xi)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((base_seed+303) % (2**63));self.q2=TeamCritic(variant,xi=xi)
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

    def probe_probabilities(self, probe):
        """Rebuild the same fixed probe prefix; hidden never leaks between calls."""
        hidden=None; values=[]
        with torch.no_grad():
            for index,snapshot in enumerate(probe['snapshots']):
                h,hidden=encode(self.actor.encoder,snapshot,hidden)
                if index in probe['indices'] and snapshot.action_mask.any():
                    values.append(self.actor.probabilities(h,snapshot)[snapshot.action_mask])
        if not values:raise ValueError('Fixed KL probe has no eligible actor decisions')
        return torch.cat(values,dim=0)

    def guarded_actor_step(self,probe,max_kl=.01,max_attempts=3):
        old_probs=self.probe_probabilities(probe).detach().clamp_min(1e-8)
        params=copy.deepcopy(self.actor.state_dict()); opt=copy.deepcopy(self.actor_optimizer.state_dict())
        initial_lr=[g['lr'] for g in self.actor_optimizer.param_groups]
        result={'accepted':False,'attempts':0,'kl':None,'step_scale':0.};attempt_kls=[]
        for attempt in range(max_attempts):
            self.actor.load_state_dict(params);self.actor_optimizer.load_state_dict(opt)
            for g,lr in zip(self.actor_optimizer.param_groups,initial_lr):g['lr']=lr*(.5**attempt)
            self.actor_optimizer.step()
            new_probs=self.probe_probabilities(probe).detach().clamp_min(1e-8)
            kl=float((new_probs*(new_probs.log()-old_probs.log())).sum(-1).mean())
            attempt_kls.append(kl if math.isfinite(kl) else None)
            result={'accepted':math.isfinite(kl) and kl<=max_kl,'attempts':attempt+1,'kl':kl if math.isfinite(kl) else None,'step_scale':.5**attempt,'attempt_kls':list(attempt_kls)}
            if result['accepted']:
                for g,lr in zip(self.actor_optimizer.param_groups,initial_lr):g['lr']=lr
                return result
        self.actor.load_state_dict(params);self.actor_optimizer.load_state_dict(opt)
        return result

    def update(self, sequences: list[tuple[list,list]], *, actor_enabled: bool=True, probe=None) -> dict:
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
                        next_probs=self.actor.probabilities(ah[t+1].detach(),tr['next_snapshot'])
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
                if actor_enabled and mask.any():
                    probs=self.actor.probabilities(ah[t],snapshot);other=sample_actions(probs.detach(),mask)
                    indices=torch.where(mask)[0]; count=len(indices)*ACTION_DIM
                    alternatives=other.repeat(count,1)
                    for k,i in enumerate(indices.tolist()): alternatives[k*ACTION_DIM:(k+1)*ACTION_DIM,i]=torch.arange(ACTION_DIM)
                    masks=mask.expand(count,-1)
                    with torch.no_grad():
                        v1=self.q1.head(q1h[t].detach().expand(count,-1,-1),alternatives,masks)[0]
                        v2=self.q2.head(q2h[t].detach().expand(count,-1,-1),alternatives,masks)[0]
                        values=torch.minimum(v1,v2).reshape(len(indices),ACTION_DIM)
                    p=probs[indices]; logp=torch.log(p.clamp_min(1e-8))
                    actor_terms.append((p*(self.alpha*logp-values)).sum()/UAV_COUNT)
                    entropies.append(-(p.detach()*logp.detach()).sum(-1).mean())
        if not critic_terms: raise ValueError('Empty sequence batch')
        critic_loss=torch.stack(critic_terms).mean()
        self.critic_optimizer.zero_grad(set_to_none=True);critic_loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for net in (self.q1,self.q2) for p in net.parameters()):
            raise FloatingPointError('Non-finite critic gradient')
        critic_encoder_gradients=[sum(p.grad.norm().item() for p in q.encoder.parameters() if p.grad is not None) for q in (self.q1,self.q2)]
        critic_conditioner_gradients=[]
        for q in (self.q1,self.q2):
            module=getattr(q.encoder,'conditioner',None)
            critic_conditioner_gradients.append(sum(p.grad.norm().item() for p in module.parameters() if p.grad is not None) if module is not None else None)
        self.critic_optimizer.step()
        actor_loss=None; conditioner_gradient=None;actor_encoder_gradient=None;residual_gradient=None;guard=None
        if actor_terms:
            actor_loss=torch.stack(actor_terms).mean();self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.actor.parameters()):
                raise FloatingPointError('Non-finite actor gradient')
            actor_encoder_gradient=sum(p.grad.norm().item() for p in self.actor.encoder.parameters() if p.grad is not None)
            residual_gradient=sum(p.grad.norm().item() for p in self.actor.residual_head.parameters() if p.grad is not None)
            conditioner=getattr(self.actor.encoder,'conditioner',None)
            if conditioner is not None:
                conditioner_gradient=float(sum(p.grad.norm().item() for p in conditioner.parameters() if p.grad is not None))
            if probe is None:raise ValueError('Actor updates require the fixed KL probe')
            guard=self.guarded_actor_step(probe)
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
                'actor_encoder_gradient_norm':actor_encoder_gradient,'residual_gradient_norm':residual_gradient,
                'critic_encoder_gradient_norms':critic_encoder_gradients,'critic_conditioner_gradient_norms':critic_conditioner_gradients,
                'entropy':torch.stack(entropies).mean().item() if entropies else None,
                'actor_guard':guard}
