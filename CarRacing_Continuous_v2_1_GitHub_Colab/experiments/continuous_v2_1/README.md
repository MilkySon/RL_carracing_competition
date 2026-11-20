# CarRacing Continuous v2.1 — isolated experiments

This package is an **overlay** for `CarRacing_A2C_Legacy/experiments/continuous_ablation/baseline`. It never changes original professor code or E0–E3 `legacy_extended.py`. Unique agent names: `e1_v2`, `e2_v2`, `e3_v2`, `sac_v2`, `td3_v2`. Do not reuse prior output directories or old checkpoints.

## 0. Install / Colab

Unzip this folder alongside the old repository. Set paths:

```python
from pathlib import Path
import subprocess,sys
V2=Path('/content/CarRacing_Continuous_v2')  # adjust to extracted folder
BASE=Path('/content/RL_carracing_competition/experiments/continuous_ablation/baseline')
ROOT=Path('/content/drive/MyDrive/CarRacing_Continuous_v2')
subprocess.run([sys.executable,str(V2/'scripts/install_overlay.py'),'--baseline',str(BASE)],check=True)
subprocess.run([sys.executable,str(V2/'scripts/test_boundaries.py')],check=True)
subprocess.run([sys.executable,str(V2/'scripts/forward_probe.py'),'--baseline',str(BASE),'--out',str(ROOT/'probe.csv')],check=True)
```

**Gate:** inspect `probe.csv`. If forward gas doesn't produce greater speed than coast/gas+brake, stop and investigate environment/action mapping; do not start 300k-frame training.

## 1. Run separate experiments

```python
# First run unshaped E2 v2, then compare with shaped E2 v2 in a new folder.
AGENT='e2_v2'  # e1_v2 / e2_v2 / e3_v2 / sac_v2 / td3_v2
RUN=ROOT/AGENT/'seed_0_unshaped'
cmd=[sys.executable,str(V2/'scripts/train_v2.py'),'--baseline',str(BASE),
     '--agent',AGENT,'--seed','0','--budget','300000','--eval-every','50000',
     '--run-dir',str(RUN)]
subprocess.run(cmd,check=True)
# After validating the grayscale grass detector, optionally append '--shaping'
# and choose a DIFFERENT --run-dir such as seed_0_shaped.
```

### What changed

- E1–E3 v2: learned initial actor bias for forward motion, lower initial Gaussian standard deviation, bounded log standard deviation; PPO rollout `old_log_prob` captured at action time (in inherited `LegacyExtended`). PPO optimizer updates only after rollout. No epsilon-random override for on-policy agents.
- SAC/TD3 v2: random 2D latent actions for `warmup_steps=2000` policy decisions, stored in replay with actual executed action; SAC stochastic exploration thereafter, TD3 Gaussian exploration thereafter. Two critics, target networks, TD3 delayed actor update. All environment actions are from the agent, not a hand-coded grass/curve rule.
- Reward shaping (opt-in): grayscale visual heuristic estimates grass near the car, gradual bounded penalty after `grass_grace` consecutive agent decisions. **No early termination**. `train.csv` includes original `return` from `RecordEpisodeStatistics` and `shaped_return`; `eval.csv` remains original reward. **WARNING**: the heuristic is deliberately experimental; grayscale dark pixels are not a reliable grass detector. Validate before interpreting results or enabling it. It can penalize legitimate road driving.
- GAE test checks terminated (zero bootstrap), truncated (bootstrap only its own final observation), and continuation (advantage propagation).

### Important limitations

- SAC uses fixed alpha=0.2, not automatic entropy tuning; not a tuned benchmark. Replay is in RAM and **not serialized**. If interrupted, a resumed off-policy checkpoint would lack replay history. Use fresh run directories for strict reproducibility.
- `train.py` writes raw Return from `info['episode']['r']`. The shaped Return in this overlay is cumulative wrapper reward before the agent's `reward_scale`. The wrapper never affects evaluation or ONNX.
- Off-policy random exploration is *initial random warmup*, not ε-greedy. TD3 noise after warmup is continuous exploration; SAC samples its policy distribution.
- ONNX export must be run with the baseline `export.py` on these new checkpoint names after confirming `check_submission.py` compatibility; no claims of passed submission validation are made.
- The visual grass detector must be checked against actual frames before treating its penalty as meaningful. Use `--shaping` only as a separately labeled ablation; keep original reward as the primary comparison.
- 300k raw frames with image-based SAC/TD3 may be insufficient; report sample efficiency rather than promising a score.


## Colab notebooks

Run `notebooks/00_Forward_And_Safety_Check.ipynb` first. It runs the forward-action diagnostic and GAE tests, without learning.
Then use `notebooks/01_E1_E2_E3_Fixes.ipynb` or `notebooks/02_SAC_TD3_Replay.ipynb`. The notebooks install the overlay only in the Colab clone and preserve all original E0–E3 files.
`notebooks/03_Grass_Shaping_Ablation.ipynb` runs an explicitly opt-in shaping ablation, **only after manually verifying** the grayscale grass heuristic.

### Important evaluation safeguards
- Do not run any `--shaping` experiment until you visually validate the heuristic on representative road/grass screenshots. Grayscale thresholding is NOT a validated grass detector.
- `old_log_prob` is saved during rollout in the *existing* `LegacyExtended.step` and checked against frozen parameters before PPO update. A2C/PPO updates only after each rollout.
- Terminated transitions have no bootstrap; truncated transitions bootstrap from the actual final observation but never propagate GAE into the next episode.
- `train.csv` raw `return` comes from original `RecordEpisodeStatistics`, while `shaped_return` is tracked separately; `eval.csv` always uses unmodified reward.
- Off-policy epsilon-random means **random latent actions** with probability epsilon (after initial warmup), not epsilon-greedy over a finite set. SAC still uses stochastic policy samples otherwise; TD3 uses Gaussian action noise otherwise.
- Replay memory is not checkpointed. Avoid `--resume` for off-policy agents; use new run dirs on Colab interruption.
- All policies require ONNX export and official `check_submission.py` validation before submission.
