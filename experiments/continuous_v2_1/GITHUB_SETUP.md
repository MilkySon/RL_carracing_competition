# Continuous v2.1 experiments (isolated branch)

This branch is reserved for the CarRacing Continuous v2.1 experiments, based on `261030`. Existing E0–E3 files should remain unchanged.

## Upload the package

Unzip `CarRacing_Continuous_v2_1.zip`, then upload the **contents** to `experiments/continuous_v2_1/` on this branch (GitHub Desktop is recommended). Keep the `agents/`, `scripts/`, and `notebooks/` directories intact. Do not upload just the ZIP.

## Colab

Open any notebook from `experiments/continuous_v2_1/notebooks/` in Colab. Use the setup cell below instead of the original ZIP-upload setup cell:

```python
from pathlib import Path
import subprocess, sys
from google.colab import drive

drive.mount('/content/drive')
REPO = Path('/content/RL_carracing_competition')
BRANCH = 'experiments/continuous-v2-1'
if not (REPO / '.git').exists():
    subprocess.run(['git','clone','--branch',BRANCH,
        'https://github.com/MilkySon/RL_carracing_competition.git',str(REPO)],check=True)
else:
    subprocess.run(['git','-C',str(REPO),'fetch','origin',BRANCH],check=True)
    subprocess.run(['git','-C',str(REPO),'checkout',BRANCH],check=True)
    subprocess.run(['git','-C',str(REPO),'pull','--ff-only','origin',BRANCH],check=True)
BASE = REPO/'experiments'/'continuous_ablation'/'baseline'
V2 = REPO/'experiments'/'continuous_v2_1'
ROOT = Path('/content/drive/MyDrive/CarRacing_Continuous_v2_1')
ROOT.mkdir(parents=True,exist_ok=True)
assert (BASE/'train.py').exists(), 'Baseline not found'
assert (V2/'scripts'/'test_boundaries.py').exists(), 'v2.1 package not uploaded to GitHub yet'
print('Code:', V2)
print('Persistent outputs:', ROOT)
```

The notebooks shipped in the original ZIP contain a ZIP upload setup cell; **replace that cell** with the above. Execute the remaining cells normally. The repository code under `/content` is ephemeral; training outputs under `/content/drive/MyDrive` persist.

Before full training, run `00_Forward_And_Safety_Check.ipynb`. The original package has not been validated by a full 300k-frame Colab run.
