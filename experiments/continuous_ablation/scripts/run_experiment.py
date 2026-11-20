"""Colab runner: safe run dirs, checkpoint export, check_submission, best/worst videos."""
import argparse,subprocess,sys,json,os,shutil
from pathlib import Path

def call(*args,cwd=None):
    print('+',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['E0','E1','E2','E3'],required=True)
    p.add_argument('--team',default='Reinforcement_material');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--budget',type=int,default=300000);p.add_argument('--eval-every',type=int,default=50000)
    p.add_argument('--root',default=None);p.add_argument('--skip-train',action='store_true')
    a=p.parse_args();base=Path(__file__).resolve().parents[1];project=base/'baseline'
    root=Path(a.root).resolve() if a.root else base/'outputs'
    run=root/a.experiment/f'seed_{a.seed}';run.mkdir(parents=True,exist_ok=True)
    agent={'E0':'a2c','E1':'e1_gae','E2':'e2_ppo','E3':'e3_coupled'}[a.experiment]
    if not a.skip_train:
        cmd=[sys.executable,'train.py','--agent',agent,'--seed',str(a.seed),'--budget',str(a.budget),'--eval-every',str(a.eval_every),'--run-dir',str(run)]
        if (run/'latest.pt').exists():cmd.append('--resume')
        elif a.experiment=='E0':cmd+=['--cfg','continuous=True']
        call(*cmd,cwd=project)
    ckpt=run/'best.pt'
    if not ckpt.exists():raise RuntimeError(f'No best.pt: {ckpt}')
    reports=run/'reports';reports.mkdir(exist_ok=True)
    call(sys.executable,base/'scripts'/'diagnostics.py','--ckpt',ckpt,'--out',reports,'--frames',a.budget,cwd=project)
    ranked=json.loads((reports/'ranked_seeds.json').read_text())
    for label,seed in [('worst',ranked['worst_seed']),('best',ranked['best_seed'])]:
        video=run/'videos'/f'{label}_seed_{seed}'
        if not video.exists() or not list(video.glob('*.mp4')):
            call(sys.executable,'evaluate.py','--ckpt',ckpt,'--seeds',str(seed),'--video',video,cwd=project)
    # Export first to unique temp filename; only publish after successful official check.
    outdir=root/'submissions';outdir.mkdir(exist_ok=True)
    stem=f'{a.team}_{a.experiment}_seed{a.seed}'
    final=outdir/f'{stem}.onnx'
    if final.exists():
        n=2
        while (outdir/f'{stem}_v{n}.onnx').exists():n+=1
        final=outdir/f'{stem}_v{n}.onnx'
    tmp=outdir/(final.stem+'.pending.onnx')
    if tmp.exists():raise RuntimeError(f'Pending export exists: {tmp}')
    try:
        call(sys.executable,'export.py','--ckpt',ckpt,'--out',tmp,cwd=project)
        # Refuse publishing a policy if Python and ONNX actions disagree.
        from importlib.util import spec_from_file_location, module_from_spec
        sys.path.insert(0,str(project))
        from evaluate import load_agent
        from export import verify
        import torch
        rate,n=verify(load_agent(str(ckpt),torch.device('cpu')),str(tmp))
        if rate<0.95:raise RuntimeError(f'Python/ONNX action agreement too low: {rate:.3f} over {n} steps')
        call(sys.executable,'check_submission.py','--onnx',tmp,'--seeds','1000-1009','--json',reports/'check_submission.json',cwd=project)
        checked=json.loads((reports/'check_submission.json').read_text())
        if not checked.get('ok'):raise RuntimeError(f'Official checker rejected: {checked}')
        if final.exists():raise FileExistsError(final)
        tmp.rename(final)
    finally:
        if tmp.exists():tmp.unlink()
    print('VALIDATED ONNX:',final)
    print('VIDEOS:',run/'videos')
    print('REPORTS:',reports)
if __name__=='__main__':main()
