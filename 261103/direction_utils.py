"""TRAINING/DIAGNOSTICS ONLY. Never passed to the deployed policy.
A track-tangent estimate from simulator geometry; approximate at intersections/low speed.
"""
import numpy as np

def direction_label(env):
    try:
        base=env.unwrapped
        car=base.car.hull
        xy=np.array([car.position[0],car.position[1]],dtype=float)
        track=np.asarray([[row[2],row[3]] for row in base.track],dtype=float)
        idx=int(np.argmin(np.sum((track-xy)**2,axis=1)))
        tangent=track[(idx+2)%len(track)]-track[(idx-2)%len(track)]
        vel=np.array([car.linearVelocity[0],car.linearVelocity[1]],dtype=float)
        if np.linalg.norm(vel)<1.0 or np.linalg.norm(tangent)<1e-6: return -1
        return int(np.dot(vel,tangent)>=0)  # 1=forward, 0=reverse
    except (AttributeError,IndexError,TypeError,ValueError):
        return -1
