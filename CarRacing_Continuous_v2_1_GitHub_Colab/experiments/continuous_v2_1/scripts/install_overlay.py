"""Copy ONLY uniquely named agent modules into baseline/agents; never overwrite originals."""
from pathlib import Path
import argparse,shutil
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);a=p.parse_args()
source=Path(__file__).resolve().parents[1]/'agents';dest=Path(a.baseline)/'agents'
assert dest.is_dir(),dest
for file in sorted(source.glob('v2_*.py')):
    target=dest/file.name
    if target.exists():
        if target.read_bytes()!=file.read_bytes():
            raise RuntimeError(f'Existing different file: {target}; refusing overwrite')
        print('Already installed:',target)
    else:
        shutil.copy2(file,target);print('Installed:',target)
