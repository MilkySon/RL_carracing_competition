"""No learning. Diagnostic actions only; NEVER use this as a submitted policy."""
import argparse,csv,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=1000);p.add_argument('--steps',type=int,default=100);a=p.parse_args()
sys.path.insert(0,str(Path(a.baseline).resolve()))
from envs import make_env
rows=[]
for name,action in [('coast',[0,0,0]),('forward',[0,0.8,0]),('gas_brake',[0,0.5,0.5])]:
    env=make_env(continuous=True,train=False);env.reset(seed=a.seed);rewards=[];speeds=[]
    try:
        for _ in range(a.steps):
            _,r,t,tr,_=env.step(np.asarray(action,dtype=np.float32));rewards.append(float(r))
            car=getattr(env.unwrapped,'car',None)
            if car is not None:
                v=car.hull.linearVelocity;speeds.append(float(np.hypot(v.x,v.y)))
            if t or tr:break
    finally:env.close()
    rows.append(dict(scenario=name,steps=len(rewards),raw_return=sum(rewards),mean_speed=np.mean(speeds) if speeds else float('nan'),max_speed=max(speeds) if speeds else float('nan')))
    print(rows[-1],flush=True)
path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
