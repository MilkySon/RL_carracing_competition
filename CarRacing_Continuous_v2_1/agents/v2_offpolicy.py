"""Image-based SAC and TD3 with replay; training-only warmup random actions.
Action parameterization: [-1,1]^2 -> [steer, gas, brake], signed throttle.
No rule-based steering or recovery; action map is state-independent.
"""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from agents.base import Agent
from networks import CNNEncoder

class Replay:
    def __init__(self,capacity):self.buf=deque(maxlen=int(capacity))
    def add(self,*item):self.buf.append(item)
    def sample(self,n,device):
        indices=np.random.randint(len(self.buf),size=n)
        rows=[self.buf[i] for i in indices]
        o,a,r,no,term,trunc=zip(*rows)
        return (torch.as_tensor(np.stack(o),device=device),torch.as_tensor(np.stack(a),dtype=torch.float32,device=device),
                torch.as_tensor(r,dtype=torch.float32,device=device).unsqueeze(1),
                torch.as_tensor(np.stack(no),device=device),
                torch.as_tensor(term,dtype=torch.float32,device=device).unsqueeze(1),
                torch.as_tensor(trunc,dtype=torch.float32,device=device).unsqueeze(1))

class Actor(nn.Module):
    def __init__(self,shape,stochastic):
        super().__init__();self.enc=CNNEncoder(shape,feature_dim=256)
        self.fc=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,4 if stochastic else 2))
        self.stochastic=stochastic
    def forward(self,obs):
        x=self.fc(self.enc(obs))
        if self.stochastic:
            mu,logstd=x.chunk(2,dim=-1)
            return mu,logstd.clamp(-5,1)
        return torch.tanh(x)
    def sample(self,obs,deterministic=False):
        mu,logstd=self(obs)
        u=mu if deterministic else Normal(mu,logstd.exp()).rsample()
        action=torch.tanh(u)
        lp=(Normal(mu,logstd.exp()).log_prob(u)-torch.log(1-action.square()+1e-6)).sum(-1,keepdim=True)
        return action,lp

class Critic(nn.Module):
    def __init__(self,shape):
        super().__init__();self.enc=CNNEncoder(shape,feature_dim=256)
        self.fc=nn.Sequential(nn.Linear(258,256),nn.ReLU(),nn.Linear(256,1))
    def forward(self,o,a):return self.fc(torch.cat([self.enc(o),a],dim=-1))

class ActionExport(nn.Module):
    def __init__(self,actor,stochastic):super().__init__();self.actor=actor;self.stochastic=stochastic
    def forward(self,obs):
        if self.stochastic:
            mu,_=self.actor(obs);a=torch.tanh(mu)
        else:a=self.actor(obs)
        x=a[:,1];return torch.stack([a[:,0],F.relu(x),F.relu(-x)],dim=-1)

class OffPolicyV2(Agent):
    default_config={'continuous':True,'lr':3e-4,'gamma':0.99,'tau':0.005,'batch_size':64,
                    'replay_capacity':4000,'warmup_steps':2000,'update_every':2,
                    'reward_scale':0.1,'policy_noise':0.2,'noise_clip':0.5,
                    'exploration_noise':0.2,'policy_delay':2,'alpha':0.2,
                    'epsilon_start':0.20,'epsilon_end':0.02,'epsilon_decay_steps':20000}
    stochastic=False
    def __init__(self,obs_shape,action_space,config,device):
        super().__init__(obs_shape,action_space,config,device)
        self.actor=Actor(obs_shape,self.stochastic).to(device)
        self.q1=Critic(obs_shape).to(device);self.q2=Critic(obs_shape).to(device)
        self.tq1=copy.deepcopy(self.q1);self.tq2=copy.deepcopy(self.q2)
        self.ta=copy.deepcopy(self.actor)
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=self.cfg['lr'])
        self.qo=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=self.cfg['lr'])
        self.replay=Replay(self.cfg['replay_capacity']);self.steps=0;self.updates=0
        self.last_latent=None;self.random_count=0
    def act(self,obs,deterministic=False):
        if not deterministic and (self.steps<self.cfg['warmup_steps'] or
            np.random.rand() < (self.cfg['epsilon_end'] +
                (self.cfg['epsilon_start']-self.cfg['epsilon_end'])*
                max(0.,1.-max(0,self.steps-self.cfg['warmup_steps'])/max(1,self.cfg['epsilon_decay_steps'])))):
            # Random exploration in latent action space; replay stores exactly this executed latent action.
            a=np.random.uniform(-1,1,size=2).astype(np.float32)
            self.random_count+=1
        else:
            with torch.no_grad():
                o=torch.as_tensor(obs,device=self.device).unsqueeze(0)
                if self.stochastic:a=self.actor.sample(o,deterministic)[0].squeeze(0).cpu().numpy()
                else:a=self.actor(o).squeeze(0).cpu().numpy()
            if not deterministic and not self.stochastic:
                a=np.clip(a+np.random.normal(0,self.cfg['exploration_noise'],size=2),-1,1).astype(np.float32)
        self.last_latent=np.asarray(a,dtype=np.float32)
        return np.asarray([a[0],max(0.,a[1]),max(0.,-a[1])],dtype=np.float32)
    def step(self,obs,action,reward,next_obs,terminated,truncated):
        self.replay.add(np.array(obs,copy=True),self.last_latent.copy(),float(reward),np.array(next_obs,copy=True),terminated,truncated)
        self.steps+=1
        if self.steps<self.cfg['warmup_steps'] or len(self.replay)<self.cfg['batch_size'] or self.steps%self.cfg['update_every']:return {}
        return self.update()
    def update(self):
        c=self.cfg;dev=self.device;self.updates+=1
        o,a,r,no,term,trunc=self.replay.sample(c['batch_size'],dev)
        # truncated bootstraps its actual final observation; terminated does not.
        with torch.no_grad():
            if self.stochastic:
                na,nlp=self.actor.sample(no)
                target=torch.minimum(self.tq1(no,na),self.tq2(no,na))-c['alpha']*nlp
            else:
                na=self.ta(no)+torch.clamp(torch.randn_like(a)*c['policy_noise'],-c['noise_clip'],c['noise_clip'])
                na=na.clamp(-1,1);target=torch.minimum(self.tq1(no,na),self.tq2(no,na))
            y=r*c['reward_scale']+c['gamma']*(1-term)*target
        qloss=F.mse_loss(self.q1(o,a),y)+F.mse_loss(self.q2(o,a),y)
        self.qo.zero_grad();qloss.backward();self.qo.step()
        logs={'q_loss':float(qloss.item()),'replay_size':len(self.replay),'random_actions':self.random_count}
        if self.stochastic or self.updates%c['policy_delay']==0:
            if self.stochastic:
                pa,lp=self.actor.sample(o)
                aloss=(c['alpha']*lp-torch.minimum(self.q1(o,pa),self.q2(o,pa))).mean()
            else:aloss=-self.q1(o,self.actor(o)).mean()
            self.ao.zero_grad();aloss.backward();self.ao.step()
            logs['actor_loss']=float(aloss.item())
            if not self.stochastic:
                self.soft(self.ta,self.actor)
        self.soft(self.tq1,self.q1);self.soft(self.tq2,self.q2)
        return logs
    def soft(self,target,src):
        with torch.no_grad():
            for tp,p in zip(target.parameters(),src.parameters()):tp.mul_(1-self.cfg['tau']).add_(p,alpha=self.cfg['tau'])
    def export_policy(self):return ActionExport(self.actor,self.stochastic)
    def state_dict(self):
        return dict(actor=self.actor.state_dict(),q1=self.q1.state_dict(),q2=self.q2.state_dict(),
                    tq1=self.tq1.state_dict(),tq2=self.tq2.state_dict(),ta=self.ta.state_dict(),
                    ao=self.ao.state_dict(),qo=self.qo.state_dict(),steps=self.steps,updates=self.updates,
                    random_count=self.random_count)
    def load_state_dict(self,s):
        for k in ('actor','q1','q2','tq1','tq2','ta'):getattr(self,k).load_state_dict(s[k])
        for k in ('ao','qo'):getattr(self,k).load_state_dict(s[k])
        self.steps=s['steps'];self.updates=s['updates'];self.random_count=s.get('random_count',0)
        # Replay not restored: avoid claiming exact resume; run new experiments instead.

class TD3V2(OffPolicyV2):name='td3_v2'
class SACV2(OffPolicyV2):name='sac_v2';stochastic=True
