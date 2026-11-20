"""Deterministic unit tests for GAE bootstrap and episode boundaries."""
import numpy as np

def gae(rewards, values, next_values, terminated, truncated, gamma=.9, lam=.95):
    out=np.zeros(len(rewards),dtype=np.float64);carry=0.
    for i in reversed(range(len(rewards))):
        delta=rewards[i]+gamma*(1-terminated[i])*next_values[i]-values[i]
        carry=delta+gamma*lam*(1-terminated[i])*(1-truncated[i])*carry
        out[i]=carry
    return out

def test():
    # terminal: no bootstrap, no cross-episode GAE
    x=gae([1.,7.],[0.,0.],[100.,100.],[1,0],[0,0])
    assert abs(x[0]-1)<1e-9,x
    # truncation: bootstrap its next state, but don't carry next episode
    x=gae([1.,7.],[0.,0.],[10.,100.],[0,0],[1,0])
    assert abs(x[0]-10)<1e-9,x
    # ordinary transition: carries next advantage
    x=gae([1.,2.],[0.,0.],[0.,0.],[0,1],[0,0])
    assert abs(x[0]-(1+.9*.95*2))<1e-9,x
    print('PASS: termination / truncation / continuation GAE tests')
if __name__=='__main__':test()


def verify_implementation(baseline):
    from pathlib import Path
    source=(Path(baseline)/'agents'/'legacy_extended.py').read_text(encoding='utf-8')
    assert 'self._last_log_prob' in source and 'old_lp' in source, 'Missing behavior log probabilities'
    assert '(1-trunc[i])' in source, 'Missing GAE truncation boundary'
    assert '(1-term)' in source, 'Missing terminal bootstrap mask'
    assert "c['reward_scale']" in source, 'Missing reward scale'
    print('PASS: source checks for stored old_log_prob, terminal/truncation masks, reward scale')
