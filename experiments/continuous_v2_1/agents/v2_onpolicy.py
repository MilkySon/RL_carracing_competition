"""Versioned on-policy fixes. Install alongside, never overwrite, legacy agents."""
import numpy as np
import torch
from agents.legacy_extended import LegacyExtended, CoupledNet, map_action

class _V2OnPolicy(LegacyExtended):
    default_config={**LegacyExtended.default_config,
        'init_gas_bias':1.2, 'init_brake_bias':-2.0,
        'init_log_std':-1.0, 'min_log_std':-2.5, 'max_log_std':0.3,
        'reward_scale':0.1}
    def __init__(self,obs_shape,action_space,config,device):
        super().__init__(obs_shape,action_space,config,device)
        with torch.no_grad():
            # New initialization is a learned policy parameter, not a per-state action override.
            last=self.net.actor.net[-1]
            if self.cfg['coupled_action']:
                last.bias[1]=self.cfg['init_gas_bias']
            else:
                last.bias[1]=self.cfg['init_gas_bias']
                last.bias[2]=self.cfg['init_brake_bias']
            self.net.log_std.fill_(self.cfg['init_log_std'])
        self._last_log_prob=None
    def act(self,obs,deterministic=False):
        with torch.no_grad():
            dist,_=self.net(torch.as_tensor(obs,device=self.device).unsqueeze(0))
            u=dist.mean if deterministic else dist.sample()
            self._last_u=u.squeeze(0).cpu().numpy()
            self._last_log_prob=float(dist.log_prob(u).item())
        return map_action(self._last_u,self.cfg['coupled_action'])
    def update(self):
        result=super().update()
        with torch.no_grad():
            self.net.log_std.clamp_(self.cfg['min_log_std'],self.cfg['max_log_std'])
        result['std_steer']=float(self.net.log_std[0].exp().item())
        return result

class E1V2(_V2OnPolicy):
    name='e1_v2'
    default_config={**_V2OnPolicy.default_config,'use_ppo':False,'coupled_action':False}
class E2V2(_V2OnPolicy):
    name='e2_v2'
    default_config={**_V2OnPolicy.default_config,'use_ppo':True,'coupled_action':False}
class E3V2(_V2OnPolicy):
    name='e3_v2'
    default_config={**_V2OnPolicy.default_config,'use_ppo':True,'coupled_action':True}
