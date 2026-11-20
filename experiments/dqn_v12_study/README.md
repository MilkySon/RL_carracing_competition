# DQN v12 study workspace

This directory is an isolated workspace for DQN v12 replication and analysis. The original experiments are unchanged.

## Colab startup

Run this in a fresh Colab notebook:

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!git clone --branch feature/dqn-v12-study --single-branch https://github.com/MilkySon/RL_carracing_competition.git
%cd /content/RL_carracing_competition/experiments/dqn_v12_study

from pathlib import Path
DRIVE_OUT = Path("/content/drive/MyDrive/RL_carracing_competition/dqn_v12_study")
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
print("Source:", Path.cwd())
print("Persistent outputs:", DRIVE_OUT)
```

On a new runtime, clone again. To refresh an existing clean clone, run `git pull --ff-only` in the repository root.

- Source code: GitHub working copy in `/content/RL_carracing_competition`
- Checkpoints, evaluation videos and logs: `/content/drive/MyDrive/RL_carracing_competition/dqn_v12_study`
- Do not commit large checkpoints or credentials to Git.
- Training code has **not** been added yet; this is a workspace/bootstrap only.
- Competition rules prohibit handwritten action overrides. Do not use low-speed forced acceleration or high-speed action suppression in a submitted policy without explicit rule clarification.
