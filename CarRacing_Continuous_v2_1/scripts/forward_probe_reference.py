"""Diagnostic only. Never include hand-coded actions in a submitted policy."""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from envs import make_env, get_frame_counter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--out',default='/content/forward_probe.csv')
    args=p.parse_args()
    # Diagnostics are not training, use unshaped environment and no training budget.
    env=make_env(continuous=True,train=False)
    scenarios={
        'coast': np.array([0.,0.,0.],dtype=np.float32),
        'forward': np.array([0.,0.8,0.],dtype=np.float32),
        'gas_and_brake':np.array([0.,0.5,0.5],dtype=np.float32),
    }
    rows=[]
    try:
        for name,action in scenarios.items():
            obs,_=env.reset(seed=args.seed)
            raw_return=0.;n=0;end_reason='step_limit';speeds=[]
            for i in range(args.steps):
                obs,reward,terminated,truncated,info=env.step(action)
                raw_return+=float(reward);n+=1
                car=getattr(env.unwrapped,'car',None)
                if car is not None:
                    v=car.hull.linearVelocity
                    speeds.append(float(np.hypot(v.x,v.y)))
                if terminated or truncated:
                    end_reason='terminated' if terminated else 'truncated'
                    break
            rows.append(dict(scenario=name,seed=args.seed,steps=n,raw_return=raw_return,
                             avg_reward=raw_return/n,mean_speed=float(np.mean(speeds)) if speeds else float('nan'),
                             max_speed=float(np.max(speeds)) if speeds else float('nan'),end_reason=end_reason,
                             frames=get_frame_counter(env).frames))
            print(rows[-1],flush=True)
    finally:
        env.close()
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Saved:',path)

if __name__=='__main__':main()
