"""E11: evaluate with separate direction diagnostics; no rule-based actions."""
import argparse,csv,json,os
import numpy as np
from evaluate import load_agent
from envs import make_env
from direction_utils import direction_label
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True);p.add_argument('--out',required=True);p.add_argument('--seeds',default='1000-1009');a=p.parse_args()
    agent=load_agent(a.ckpt,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True)
    rows=[]
    lo,hi=map(int,a.seeds.split('-'))
    for seed in range(lo,hi+1):
        env=make_env(continuous=False,train=False);obs,_=env.reset(seed=seed)
        reverse_steps=0;streak=0;max_streak=0;steps=0;done=False
        while not done:
            label=direction_label(env)
            if label==0:reverse_steps+=1;streak+=1;max_streak=max(max_streak,streak)
            else:streak=0
            obs,_,terminated,truncated,info=env.step(agent.act(obs,deterministic=True))
            done=terminated or truncated;steps+=1
        rows.append({'seed':seed,'return':float(info['episode']['r']),'agent_steps':steps,'reverse_steps':reverse_steps,'max_reverse_streak':max_streak})
        env.close()
    with open(a.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
