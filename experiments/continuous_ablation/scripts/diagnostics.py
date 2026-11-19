"""Evaluation-only diagnostics. Never used for action selection or reward shaping.
Offroad and curve labels are approximate track-center geometry proxies, NOT official metrics.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
from evaluate import load_agent
from envs import make_env,FRAME_SKIP,get_frame_counter

SEEDS=list(range(1000,1010))
def get_geometry(env):
    u=env.unwrapped
    track=getattr(u,'track',None)
    car=getattr(u,'car',None)
    if not track or car is None:return None
    pts=np.array([[p[2],p[3]] for p in track],dtype=float)
    pos=np.array(car.hull.position,dtype=float)
    d=np.linalg.norm(pts-pos,axis=1); i=int(d.argmin())
    # Track width is 40/SCALE (gymnasium CarRacing); half width threshold is proxy only.
    from gymnasium.envs.box2d import car_racing
    half_width=float(car_racing.TRACK_WIDTH)/float(car_racing.SCALE)/2
    a=pts[(i-3)%len(pts)]-pts[i];b=pts[(i+3)%len(pts)]-pts[i]
    ang=math.acos(float(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9),-1,1)))
    curve=(math.pi-ang)>0.16 # fixed threshold; geometry proxy
    return dict(offroad=bool(d[i]>half_width),curve=bool(curve),tile_count=int(getattr(u,'tile_visited_count',0)),track_tiles=len(track))
def one(agent,seed,trace=False):
    env=make_env(continuous=True,train=False)
    obs,_=env.reset(seed=seed)
    done=False;overlap=0;overlap_frames=0;total_frames=0;off_events=0;recoveries=0;curve_entries=0;curve_fails=0
    was_off=False;was_curve=False;off_on_curve=False;events=[];actions=[];t=0
    counter=get_frame_counter(env)
    while not done:
        before=counter.frames
        a=agent.act(obs,deterministic=True)
        nxt,r,terminated,truncated,info=env.step(a)
        t+=1;# each action is repeated for up to FRAME_SKIP raw frames
        total_frames+=FRAME_SKIP
        overlap+=int(a[1]>0.1 and a[2]>0.1)
        geo=get_geometry(env)
        if geo:
            off=geo['offroad'];curve=geo['curve']
            if curve and not was_curve:curve_entries+=1
            if off and not was_off:
                off_events+=1;off_on_curve=curve
                if curve:curve_fails+=1
                events.append(dict(seed=seed,step=t,event='offroad',curve=int(curve)))
            if not off and was_off:
                recoveries+=1;events.append(dict(seed=seed,step=t,event='recovery',curve=int(curve)))
            was_off=off;was_curve=curve
        if trace:actions.append(dict(seed=seed,step=t,steer=float(a[0]),gas=float(a[1]),brake=float(a[2]),offroad=int(geo['offroad']) if geo else '',curve=int(geo['curve']) if geo else ''))
        obs=nxt;done=terminated or truncated
    u=env.unwrapped
    completed=bool(getattr(u,'tile_visited_count',0)>=len(getattr(u,'track',[]) or [0]))
    result=dict(seed=seed,return_raw=float(info['episode']['r']),completed=int(completed),offroad_events=off_events,recoveries=recoveries,curve_entries=curve_entries,curve_failures=curve_fails,overlap_steps=overlap,overlap_frames=overlap_frames,raw_frames=total_frames,action_steps=t)
    env.close();return result,events,actions

def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True);p.add_argument('--out',required=True);p.add_argument('--seeds',default='1000-1009');p.add_argument('--frames',type=int,default=0)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    seeds=list(range(int(a.seeds.split('-')[0]),int(a.seeds.split('-')[1])+1)) if '-' in a.seeds else [int(x) for x in a.seeds.split(',')]
    agent=load_agent(a.ckpt,torch.device('cpu'))
    episodes=[];events=[];actions=[]
    for seed in seeds:
        ep,ev,ac=one(agent,seed,trace=True);episodes.append(ep);events.extend(ev);actions.extend(ac)
        print('seed',seed,'return',round(ep['return_raw'],2),'completed',ep['completed'])
    write_csv(out/'eval_episodes.csv',episodes);write_csv(out/'eval_events.csv',events);write_csv(out/'action_trace.csv',actions)
    s=lambda k:sum(x[k] for x in episodes)
    summary=dict(frames=a.frames,mean_return=float(np.mean([x['return_raw'] for x in episodes])),std_return=float(np.std([x['return_raw'] for x in episodes])),completion_rate=s('completed')/len(episodes),offroad_recovery_rate=s('recoveries')/s('offroad_events') if s('offroad_events') else None,gas_brake_overlap_rate=s('overlap_frames')/s('raw_frames') if s('raw_frames') else None,curve_failure_rate=s('curve_failures')/s('curve_entries') if s('curve_entries') else None,metric_note='offroad/curve are geometric proxies; overlap is raw-frame weighted; no offroad events -> null')
    (out/'eval_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    (out/'ranked_seeds.json').write_text(json.dumps({'worst_seed':min(episodes,key=lambda x:x['return_raw'])['seed'],'best_seed':max(episodes,key=lambda x:x['return_raw'])['seed']},indent=2))
if __name__=='__main__':main()
