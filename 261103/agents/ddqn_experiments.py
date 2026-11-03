"""E10 Double DQN; E11 same policy with external diagnostics;
E12 Double DQN + negative-reward-focused replay; E13 + train-only direction auxiliary head.
No hand-coded steering. All submitted actions are learned Q-network outputs.
"""
import numpy as np
import torch
import torch.nn.functional as F
from buffers import ReplayBuffer
from networks import QNetwork
from .dqn import DQNAgent

class DDQNAgent(DQNAgent):
    name='ddqn_e10'
    def update(self,batch):
        c=self.cfg
        q=self.q(batch['obs']).gather(1,batch['actions'].long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            chosen=self.q(batch['next_obs']).argmax(dim=1,keepdim=True)
            nxt=self.q_target(batch['next_obs']).gather(1,chosen).squeeze(1)
            target=batch['rewards']+c['gamma']*(1.-batch['dones'])*nxt
        loss=F.smooth_l1_loss(q,target)
        self.optim.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(),c['max_grad_norm']);self.optim.step()
        self.updates+=1
        if self.updates%c['target_update_interval']==0:self.q_target.load_state_dict(self.q.state_dict())
        return {'loss':float(loss.item()),'q_mean':float(q.mean().item())}

class DiagnosticDDQN(DDQNAgent):
    name='ddqn_e11'

class FailureReplayDDQN(DDQNAgent):
    name='ddqn_e12'
    default_config={**DQNAgent.default_config,'failure_fraction':0.25,'failure_buffer_size':10000}
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.failure_buffer=None
    def step(self,obs,action,reward,next_obs,terminated,truncated):
        if self.buffer is None:self.buffer=ReplayBuffer(self.cfg['buffer_size'],self.obs_shape)
        self.buffer.add(obs,action,reward,next_obs,terminated)
        # Proxy: negative immediate reward is a failure-related transition, NOT a verified reverse label.
        if reward<0:
            if self.failure_buffer is None:self.failure_buffer=ReplayBuffer(self.cfg['failure_buffer_size'],self.obs_shape)
            self.failure_buffer.add(obs,action,reward,next_obs,terminated)
        self.steps+=1;c=self.cfg
        if len(self.buffer)<c['learning_starts'] or self.steps%c['train_freq']:return {}
        n=int(c['batch_size']*c['failure_fraction']) if self.failure_buffer is not None and len(self.failure_buffer)>0 else 0
        regular=self.buffer.sample(c['batch_size']-n,self.device)
        if n:
            focused=self.failure_buffer.sample(n,self.device)
            regular={k:torch.cat([v,focused[k]],dim=0) for k,v in regular.items()}
        m=self.update(regular);m['epsilon']=self.epsilon();m['failure_samples']=n
        return m

class DirectionQNetwork(QNetwork):
    def __init__(self,obs_shape,n_actions):
        super().__init__(obs_shape,n_actions)
        self.direction_head=torch.nn.Linear(self.encoder.feature_dim,2)
    def direction_logits(self,obs):
        return self.direction_head(self.encoder(obs))

class DirectionAuxDDQN(FailureReplayDDQN):
    name='ddqn_e13'
    default_config={**FailureReplayDDQN.default_config,'aux_weight':0.1}
    def __init__(self,obs_shape,action_space,config,device):
        super().__init__(obs_shape,action_space,config,device)
        self.q=DirectionQNetwork(obs_shape,self.n_actions).to(device)
        self.q_target=DirectionQNetwork(obs_shape,self.n_actions).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim=torch.optim.Adam(self.q.parameters(),lr=self.cfg['lr'])
        self.label_buffer=None
    def step(self,obs,action,reward,next_obs,terminated,truncated):
        # direction_label is supplied only by the training loop; ignored at evaluation/export.
        return super().step(obs,action,reward,next_obs,terminated,truncated)
    def update(self,batch):
        c=self.cfg
        q=self.q(batch['obs']).gather(1,batch['actions'].long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a=self.q(batch['next_obs']).argmax(1,keepdim=True)
            next_q=self.q_target(batch['next_obs']).gather(1,a).squeeze(1)
            y=batch['rewards']+c['gamma']*(1-batch['dones'])*next_q
        td=F.smooth_l1_loss(q,y)
        aux=torch.zeros((),device=self.device)
        if self.label_buffer is not None and len(self.label_buffer)>0:
            k=min(c['batch_size'],len(self.label_buffer))
            ids=np.random.randint(len(self.label_buffer),size=k)
            samples=torch.as_tensor(np.stack([self.label_buffer[i][0] for i in ids]),device=self.device)
            labels=torch.as_tensor([self.label_buffer[i][1] for i in ids],device=self.device,dtype=torch.long)
            aux=F.cross_entropy(self.q.direction_logits(samples),labels)
        loss=td+c['aux_weight']*aux
        self.optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.q.parameters(),c['max_grad_norm']);self.optim.step()
        self.updates+=1
        if self.updates%c['target_update_interval']==0:self.q_target.load_state_dict(self.q.state_dict())
        return {'loss':float(td.item()),'aux_loss':float(aux.item()),'q_mean':float(q.mean().item())}
    def add_direction_example(self,obs,label):
        if label not in (0,1):return
        from collections import deque
        if self.label_buffer is None:self.label_buffer=deque(maxlen=2000)
        self.label_buffer.append((np.asarray(obs,dtype=np.uint8).copy(),int(label)))
