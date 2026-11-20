"""Run versioned agents using the unchanged baseline training loop.
Only monkeypatches the env factory and CSV writer in this process; baseline files untouched.
"""
import argparse,sys,csv
from pathlib import Path
import numpy as np
import gymnasium as gym

p=argparse.ArgumentParser()
p.add_argument('--baseline',required=True);p.add_argument('--agent',required=True,choices=['e1_v2','e2_v2','e3_v2','sac_v2','td3_v2'])
p.add_argument('--run-dir',required=True);p.add_argument('--seed',type=int,default=0)
p.add_argument('--budget',type=int,default=300000);p.add_argument('--eval-every',type=int,default=50000)
p.add_argument('--shaping',action='store_true');p.add_argument('--grass-threshold',type=float,default=.38)
p.add_argument('--grass-grace',type=int,default=12);p.add_argument('--grass-penalty',type=float,default=.03)
p.add_argument('--cfg',nargs='*',default=[])
a=p.parse_args()
baseline=Path(a.baseline).resolve();sys.path.insert(0,str(baseline))
import train as legacy_train
from envs import make_env as original_make_env

class TrainingGrassPenalty(gym.Wrapper):
    """Opt-in approximate grass detector from grayscale visual observation.
    NOT a certified road/grass classifier. Validate against recorded images first.
    Does not alter actions, episode termination, or original episode return.
    """
    def __init__(self,env):
        super().__init__(env);self.streak=0;self.shaped=0.;self.penalties=0.;self.count=0
    def reset(self,**kwargs):
        self.streak=0;self.shaped=0.;self.penalties=0.;self.count=0
        return self.env.reset(**kwargs)
    def step(self,action):
        obs,r,term,trunc,info=self.env.step(action)
        # Two side patches near the car. Grass tends to be dark in grayscale,
        # but shadows/road colors can be confounders; this is experimental.
        img=obs[-1]
        left=img[58:70,29:38];right=img[58:70,58:67]
        darkness=(float(np.mean(left<95))+float(np.mean(right<95)))/2
        grass=darkness>=a.grass_threshold
        self.streak=self.streak+1 if grass else 0
        penalty=0.0
        if self.streak>a.grass_grace:
            penalty=min(a.grass_penalty*(self.streak-a.grass_grace),a.grass_penalty*10)
        self.shaped+=float(r)-penalty;self.penalties+=penalty;self.count+=1
        if term or trunc:
            # Original info['episode']['r'] remains the original reward.
            info=dict(info);info['v2_shaped_return']=self.shaped
            info['v2_grass_penalty_total']=self.penalties
            last_episode['shaped_return']=self.shaped
            last_episode['grass_penalty_total']=self.penalties
        return obs,float(r)-penalty,term,trunc,info

last_episode={}
def factory(*args,**kwargs):
    env=original_make_env(*args,**kwargs)
    if kwargs.get('train',True) and a.shaping:
        return TrainingGrassPenalty(env)
    return env

orig_append=legacy_train.append_csv
def append_csv(path,row):
    if str(path).endswith('train.csv'):
        row=dict(row)
        if a.shaping:
            row.update(last_episode)
        else:
            row['shaped_return']=row['return']
            row['grass_penalty_total']=0.0
    return orig_append(path,row)
legacy_train.make_env=factory
legacy_train.append_csv=append_csv
sys.argv=['train.py','--agent',a.agent,'--seed',str(a.seed),'--budget',str(a.budget),
          '--eval-every',str(a.eval_every),'--run-dir',a.run_dir,'--cfg',*a.cfg]
print('Versioned run:',a.agent,'shaping=',a.shaping,'run_dir=',a.run_dir,flush=True)
legacy_train.main()
