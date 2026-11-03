"""Read-only raw-frame diagnostics. Does not change actions, rewards or termination."""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import gymnasium as gym
import torch
from envs import make_env, FrameSkip
from evaluate import load_agent, parse_seeds
from direction_utils import direction_label

class Observer(gym.Wrapper):
    def __init__(self, env, writer):
        super().__init__(env)
        self.writer = writer

    def reset(self, **kwargs):
        self.seed = kwargs.get('seed')
        self.frame = self.offroad_count = self.exits = self.reentries = 0
        self.reverse_count = self.reverse_streak = self.max_reverse = 0
        self.progress_recoveries = 0
        self.was_offroad = False
        self.pending_recovery = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame += 1
        base = self.unwrapped
        contacts = sum(bool(w.tiles) for w in base.car.wheels)
        offroad = contacts == 0
        tiles = base.tile_visited_count
        direction = direction_label(self)
        new_tiles = tiles - getattr(self, 'last_tiles', tiles) if self.frame > 1 else 0
        if offroad and not self.was_offroad:
            self.exits += 1
            self.pending_recovery = True
        if not offroad and self.was_offroad:
            self.reentries += 1
        recovered = self.pending_recovery and not offroad and direction == 1 and new_tiles > 0
        if recovered:
            self.progress_recoveries += 1
            self.pending_recovery = False
        self.offroad_count += int(offroad)
        self.reverse_count += int(direction == 0)
        self.reverse_streak = self.reverse_streak + 1 if direction == 0 else 0
        self.max_reverse = max(self.max_reverse, self.reverse_streak)
        self.writer.writerow(dict(seed=self.seed,raw_frame=self.frame,reward=reward,
            wheel_contacts=contacts,offroad=int(offroad),direction=direction,
            tiles=tiles,new_tiles=new_tiles,recovered=int(recovered),
            terminated=int(terminated),truncated=int(truncated)))
        self.was_offroad,self.last_tiles = offroad,tiles
        if terminated or truncated:
            reason = 'time_limit' if truncated and not terminated else 'environment_termination'
            if terminated and reward <= -100: reason = 'out_of_bounds_inferred'
            self.summary = dict(seed=self.seed,return_=float(info['episode']['r']),
                raw_frames=self.frame,offroad_frames=self.offroad_count,exits=self.exits,
                road_reentries=self.reentries,forward_progress_recoveries=self.progress_recoveries,
                reverse_frames=self.reverse_count,max_reverse_streak=self.max_reverse,
                terminated=bool(terminated),truncated=bool(truncated),end_reason=reason)
        return obs,reward,terminated,truncated,info


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',required=True)
    p.add_argument('--seeds',type=parse_seeds,default=list(range(1000,1010)))
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent=load_agent(args.ckpt,device)
    fields=['seed','raw_frame','reward','wheel_contacts','offroad','direction','tiles','new_tiles','recovered','terminated','truncated']
    summaries=[]
    with (out/'frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        env=make_env(continuous=agent.continuous,train=False)
        cursor=env
        while not isinstance(cursor,FrameSkip):cursor=cursor.env
        observer=Observer(cursor.env,writer)
        cursor.env=observer
        wrappers=[];node=env
        while isinstance(node,gym.Wrapper):
            wrappers.append(type(node).__name__);node=node.env
        try:
            for seed in args.seeds:
                obs,_=env.reset(seed=seed)
                while True:
                    obs,_,term,trunc,_=env.step(agent.act(obs,deterministic=True))
                    if term or trunc:break
                summaries.append(observer.summary);print(observer.summary)
        finally:env.close()
    with (out/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    env_file=Path(__file__).with_name('envs.py')
    audit={'checkpoint':str(Path(args.ckpt).resolve()),'seeds':args.seeds,'gymnasium':gym.__version__,
        'wrapper_chain':wrappers,'envs_sha256':hashlib.sha256(env_file.read_bytes()).hexdigest(),
        'make_env_source':inspect.getsource(make_env),
        'note':'Bundled make_env has no custom offroad termination. Direction is approximate; end_reason inferred, not an authoritative environment label. Observer runs after each raw frame; initial spawn offroad can count as an exit. Recovery has no time-window limit; review frames.csv.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
