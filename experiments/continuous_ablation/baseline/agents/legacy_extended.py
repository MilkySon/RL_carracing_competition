"""Controlled E1/E2/E3 extensions; E0 uses the untouched professor A2C."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from agents.actor_critic import ActorCriticAgent, ActorCriticNet

class CoupledNet(ActorCriticNet):
    def __init__(self, obs_shape, action_space):
        super().__init__(obs_shape, action_space)
        from networks import MLPHead
        self.actor = MLPHead(self.encoder.feature_dim, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

def map_action(u, coupled=False):
    t = np.tanh(u)
    if not coupled:
        return np.array([t[0],(t[1]+1)/2,(t[2]+1)/2],dtype=np.float32)
    x=float(t[1]); d=0.10
    return np.array([t[0],max(0.,(x-d)/(1-d)),max(0.,(-x-d)/(1-d))],dtype=np.float32)

class ExportPolicy(nn.Module):
    def __init__(self, net, coupled):
        super().__init__(); self.net=net; self.coupled=coupled
    def forward(self, obs):
        t=torch.tanh(self.net.actor(self.net.encoder(obs)))
        if not self.coupled:
            return torch.stack([t[:,0],(t[:,1]+1)/2,(t[:,2]+1)/2],dim=-1)
        x=t[:,1]; d=0.10
        return torch.stack([t[:,0],torch.relu(x-d)/(1-d),torch.relu(-x-d)/(1-d)],dim=-1)

class LegacyExtended(ActorCriticAgent):
    name='legacy_extended'
    default_config={**ActorCriticAgent.default_config, 'continuous':True,
                    'gae_lambda':0.95,'use_ppo':False,'coupled_action':False,
                    'ppo_epochs':4,'minibatch_size':64,'clip_eps':0.2,'target_kl':0.02}
    def __init__(self, obs_shape, action_space, config, device):
        super().__init__(obs_shape,action_space,config,device)
        assert self.continuous
        if self.cfg['coupled_action']:
            self.net=CoupledNet(obs_shape,action_space).to(device)
            self.optim=torch.optim.Adam(self.net.parameters(),lr=self.cfg['lr'])
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            dist,_=self.net(torch.as_tensor(obs,device=self.device).unsqueeze(0))
            u=dist.mean if deterministic else dist.sample()
        self._last_u=u.squeeze(0).cpu().numpy()
        return map_action(self._last_u,self.cfg['coupled_action'])
    def update(self):
        c=self.cfg; dev=self.device
        obs,u,r,nxt,term,trunc=zip(*self.rollout)
        obs=torch.as_tensor(np.stack(obs),device=dev)
        nxt=torch.as_tensor(np.stack(nxt),device=dev)
        u=torch.as_tensor(np.stack(u),dtype=torch.float32,device=dev)
        r=torch.as_tensor(r,dtype=torch.float32,device=dev)*c['reward_scale']
        term=torch.as_tensor(term,dtype=torch.float32,device=dev)
        trunc=torch.as_tensor(trunc,dtype=torch.float32,device=dev)
        with torch.no_grad():
            old_dist,values=self.net(obs)
            _,next_values=self.net(nxt)
            old_lp=old_dist.log_prob(u)
            delta=r+c['gamma']*(1-term)*next_values-values
            adv=torch.zeros_like(r); running=torch.zeros((),device=dev)
            for i in range(len(r)-1,-1,-1):
                # Truncation bootstraps its own next state, but does not link to reset episode.
                running=delta[i]+c['gamma']*c['gae_lambda']*(1-term[i])*(1-trunc[i])*running
                adv[i]=running
            ret=adv+values
            if c['normalize_advantage'] and len(adv)>1:
                adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        n=len(r)
        if not c['use_ppo']:
            dist,v=self.net(obs)
            pl=-(dist.log_prob(u)*adv).mean()
            vl=(v-ret).square().mean()
            ent=dist.entropy().mean()
            loss=pl+c['value_coef']*vl-c['entropy_coef']*ent
            self.optim.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),c['max_grad_norm']);self.optim.step()
            return dict(loss=float(loss.item()),policy_loss=float(pl.item()),value_loss=float(vl.item()),entropy=float(ent.item()))
        logs=[]
        for epoch in range(c['ppo_epochs']):
            order=torch.randperm(n,device=dev)
            stop=False
            for ix in order.split(c['minibatch_size']):
                dist,v=self.net(obs[ix]); lp=dist.log_prob(u[ix]); ratio=(lp-old_lp[ix]).exp()
                pl=-torch.minimum(ratio*adv[ix],torch.clamp(ratio,1-c['clip_eps'],1+c['clip_eps'])*adv[ix]).mean()
                vl=(v-ret[ix]).square().mean();ent=dist.entropy().mean()
                loss=pl+c['value_coef']*vl-c['entropy_coef']*ent
                approx_kl=((ratio-1)-(lp-old_lp[ix])).mean()
                if c['target_kl'] is not None and approx_kl.item()>c['target_kl']:
                    stop=True;break
                self.optim.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),c['max_grad_norm']);self.optim.step()
                logs.append([loss.item(),pl.item(),vl.item(),ent.item(),approx_kl.item(),((ratio-1).abs()>c['clip_eps']).float().mean().item()])
            if stop:break
        if not logs:
            return {'early_kl_stop':1.0}
        z=np.mean(logs,axis=0)
        return dict(zip(['loss','policy_loss','value_loss','entropy','approx_kl','clip_fraction'],map(float,z)))
    def export_policy(self):
        return ExportPolicy(self.net,self.cfg['coupled_action'])

class E1Agent(LegacyExtended):
    name='e1_gae'
    default_config={**LegacyExtended.default_config,'use_ppo':False,'coupled_action':False}
class E2Agent(LegacyExtended):
    name='e2_ppo'
    default_config={**LegacyExtended.default_config,'use_ppo':True,'coupled_action':False}
class E3Agent(LegacyExtended):
    name='e3_coupled'
    default_config={**LegacyExtended.default_config,'use_ppo':True,'coupled_action':True}
