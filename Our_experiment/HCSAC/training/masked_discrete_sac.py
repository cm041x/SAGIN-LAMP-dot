"""Original discrete SAC objective with explicit graph/slot masking."""
import math
import torch
from torch_geometric.data import Batch
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import PolicyNet_CNN,QValueNet_CNN,PolicyNet_GCN,QValueNet_GCN

class MaskedDiscreteSAC:
    def __init__(self, kind: str, device: str='cpu', lr: float=3e-4, alpha_lr: float=1e-4,
                 gamma: float=.99, tau: float=.005, initial_alpha: float=.01):
        if kind not in ('CNN','GCN'): raise ValueError('kind must be CNN or GCN')
        self.kind=kind;self.device=torch.device(device); self.gamma=gamma;self.tau=tau; self.actions=8 if kind=='CNN' else 5
        actor,critic=(PolicyNet_CNN,QValueNet_CNN) if kind=='CNN' else (PolicyNet_GCN,QValueNet_GCN)
        dim=15 if kind=='CNN' else 6
        self.actor=actor(dim,128,self.actions).to(self.device)
        self.critic_1=critic(dim,128,self.actions).to(self.device);self.critic_2=critic(dim,128,self.actions).to(self.device)
        self.target_critic_1=critic(dim,128,self.actions).to(self.device);self.target_critic_2=critic(dim,128,self.actions).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict());self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.critic_1_optimizer=torch.optim.Adam(self.critic_1.parameters(),lr=lr)
        self.critic_2_optimizer=torch.optim.Adam(self.critic_2.parameters(),lr=lr)
        self.log_alpha=torch.tensor(math.log(initial_alpha),dtype=torch.float32,device=self.device,requires_grad=True)
        self.alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        self.target_entropy=-math.log(self.actions); self.skipped_empty_mask_updates=0

    def load_original(self, path: str) -> None:
        ck=torch.load(path,map_location=self.device,weights_only=True)
        for key in ('actor','critic_1','critic_2','target_critic_1','target_critic_2'):
            getattr(self,key).load_state_dict(ck[key+'_state_dict'],strict=True)
        with torch.no_grad(): self.log_alpha.copy_(torch.as_tensor(ck['log_alpha'],device=self.device))

    def _graph(self, net, graphs: list) -> torch.Tensor:
        batch=Batch.from_data_list(graphs).to(self.device); mask=batch.node_type==0
        graphs_id=batch.batch[mask]; slots=batch.slot_id[mask].long()
        if len(slots)!=6*len(graphs) or not torch.equal(torch.sort(graphs_id*6+slots).values,torch.arange(6*len(graphs),device=self.device)):
            raise ValueError('Graph node/slot mapping invalid')
        outputs=net(batch); return torch.zeros((len(graphs)*6,self.actions),device=self.device,dtype=outputs.dtype).index_copy(0,graphs_id*6+slots,outputs).reshape(len(graphs),6,self.actions)

    def _forward(self, net, state):
        return net(state) if self.kind=='CNN' else self._graph(net,state)

    @staticmethod
    def _check_grad(model) -> None:
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError('Non-finite gradient')

    def update(self, samples: list) -> dict:
        if self.kind=='CNN':
            import numpy as np
            x=torch.as_tensor(np.stack([s[0] for s in samples]),device=self.device)
            act=torch.tensor([s[1] for s in samples],device=self.device); rew=torch.tensor([s[2] for s in samples],device=self.device)
            nx=torch.as_tensor(np.stack([s[3] for s in samples]),device=self.device); done=torch.tensor([s[4] for s in samples],device=self.device)
            mask=torch.ones_like(done,dtype=torch.bool); nextmask=~done
        else:
            x=[s[0] for s in samples];nx=[s[3] for s in samples]
            import numpy as np
            act=torch.as_tensor(np.stack([s[1] for s in samples]),device=self.device);rew=torch.as_tensor(np.stack([s[2] for s in samples]),device=self.device)
            mask=torch.as_tensor(np.stack([s[4] for s in samples]),device=self.device)
            nextmask=torch.as_tensor(np.stack([s[5] for s in samples]),device=self.device)
            done=torch.as_tensor(np.stack([s[6] for s in samples]),device=self.device)
        if not mask.any():
            self.skipped_empty_mask_updates+=1;return {'skipped':True}
        with torch.no_grad():
            probs=self._forward(self.actor,nx); lp=torch.log(probs+1e-8)
            q=torch.minimum(self._forward(self.target_critic_1,nx),self._forward(self.target_critic_2,nx))
            value=(probs*(q-self.log_alpha.exp()*lp)).sum(-1)
            target=rew+self.gamma*value*(nextmask & ~done).float()
        losses=[]
        for net,opt in ((self.critic_1,self.critic_1_optimizer),(self.critic_2,self.critic_2_optimizer)):
            q=self._forward(net,x).gather(-1,act.unsqueeze(-1)).squeeze(-1)
            loss=((q-target)**2)[mask].mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite critic loss')
            opt.zero_grad(set_to_none=True);loss.backward();self._check_grad(net);opt.step();losses.append(loss.detach().item())
        probs=self._forward(self.actor,x);logp=torch.log(probs+1e-8)
        with torch.no_grad():q=torch.minimum(self._forward(self.critic_1,x),self._forward(self.critic_2,x))
        entropy=-(probs*logp).sum(-1)
        actor_loss=(probs*(self.log_alpha.exp().detach()*logp-q)).sum(-1)[mask].mean()
        if not torch.isfinite(actor_loss):raise FloatingPointError('Non-finite actor loss')
        self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward();self._check_grad(self.actor);self.actor_optimizer.step()
        alpha_loss=((entropy-self.target_entropy).detach()*self.log_alpha.exp())[mask].mean()
        if not torch.isfinite(alpha_loss):raise FloatingPointError('Non-finite alpha loss')
        self.alpha_optimizer.zero_grad(set_to_none=True);alpha_loss.backward()
        if not torch.isfinite(self.log_alpha.grad):raise FloatingPointError('Non-finite alpha gradient')
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target,source in ((self.target_critic_1,self.critic_1),(self.target_critic_2,self.critic_2)):
                for a,b in zip(target.parameters(),source.parameters()):a.lerp_(b,self.tau)
        return {'actor_loss':actor_loss.item(),'critic_loss':sum(losses)/2,'alpha_loss':alpha_loss.item(),
                'alpha':self.log_alpha.exp().item(),'entropy':entropy[mask].mean().item(),'skipped':False}
