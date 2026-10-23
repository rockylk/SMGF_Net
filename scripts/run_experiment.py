"""Portable path adapter for the checksum-preserved final experiment runners.

This adapter changes paths only. It does not change model, loss, attack, optimizer,
selection, seed, or epoch settings. Dry runs use the Python standard library only.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'src/smgfnet/frozen'
ENTRIES={
    'ustc_cnn':'src/baselines/ustc_cnn/USTC_CNN_final_paper_experiments.py',
    'etbert':'src/baselines/etbert/ET-BERT_final_paper_experiments.py',
    'pean_nhp':'src/baselines/pean_nhp/PEAN_NHP_final_paper_experiments_V1_100ep.py',
    'ibgc':'src/baselines/ibgc/IBGC_final_paper_experiments_V1_100ep.py',
}

def resolve(value):
    p=Path(value).expanduser()
    return (p if p.is_absolute() else ROOT/p).resolve()

def read_config(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig')) if path else {}

def settings(args):
    cfg=read_config(args.config)
    for key in ('cipher_root','ustc_root','result_root','cache_root','etbert_root','smgf_seed42_checkpoint'):
        value=getattr(args,key,None)
        if value:
            cfg[key]=value
    cfg.setdefault('result_root','outputs/RESULT')
    cfg.setdefault('cache_root','outputs/cache')
    return {k:str(resolve(v)) for k,v in cfg.items() if v}

def assert_preserved_sources():
    manifest=json.loads((ROOT/'manifests/source_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        if not row['path'].endswith('.py'):
            continue
        path=ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError(f'Archived source was modified: {row["path"]}')

def environment(args,cfg):
    env=os.environ.copy()
    env.update(SMGF_WORK_ROOT=str(ROOT),SMGF_RESULT_DIR=cfg['result_root'],
               SMGF_USTC_PILOT_FRACTION='1.0',PYTHONDONTWRITEBYTECODE='1')
    for prefix in ('USTCCNN','ETBERT','PEAN_NHP_FINAL','IBGC'):
        for ds in ('cipher','ustc'):
            env[f'{prefix}_{ds.upper()}_SPLIT_MANIFEST']=str(ROOT/f'data/splits/{ds}.json')
            if f'{ds}_root' in cfg:
                env[f'{prefix}_{ds.upper()}_ROOT']=cfg[f'{ds}_root']
    env['USTCCNN_RESULT_ROOT']=cfg['result_root']
    env['ETBERT_RESULT_ROOT']=cfg['result_root']
    env['PEAN_NHP_RESULT_ROOT']=cfg['result_root']
    env['ETBERT_FINAL_MODE']=args.mode
    env['ETBERT_FINAL_DATASETS']=args.dataset
    if cfg.get('etbert_root'):
        env['ETBERT_ROOT']=cfg['etbert_root']
    if cfg.get('cipher_root'):
        env['SMGF_PCAP_DIR']=cfg['cipher_root']
    if cfg.get('smgf_seed42_checkpoint'):
        env['SMGF_SEED42_CKPT']=cfg['smgf_seed42_checkpoint']
        env['SMGF_CIPHER_FULL_SEED42_CKPT']=cfg['smgf_seed42_checkpoint']
    dirs=[ROOT/'src/baselines/common']+[ROOT/f'src/baselines/{m}' for m in ENTRIES]+[FROZEN]
    env['PYTHONPATH']=os.pathsep.join(map(str,dirs))+os.pathsep+env.get('PYTHONPATH','')
    return env

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def relocate_run(module,new_root):
    """Relocate path globals derived from RUN_DIR, retaining all filenames."""
    old=module.RUN_DIR
    new=Path(new_root)/old.name
    for key,value in list(vars(module).items()):
        if isinstance(value,Path) and value.is_relative_to(old):
            setattr(module,key,new/value.relative_to(old))
    module.RESULT_ROOT=Path(new_root)

def native_smgf(args,cfg):
    os.environ.update(environment(args,cfg))
    if args.model=='ablation':
        module=load(FROZEN/'V17_CipherSpectrum_MATCHED_ABLATION_SEED42.py','archive_matched_ablation')
        module.CIPHER_ROOT=Path(cfg['cipher_root'])
    elif args.dataset=='cipher':
        module=load(FROZEN/'V17_CipherSpectrum_final_frozen_3seed.py','archive_cipher_final')
    else:
        os.environ['SMGF_USTC_TRAIN_SEED']=args.seeds
        module=load(FROZEN/'V17_USTC_frozen_runner.py','archive_ustc_final')
        module.USTC_ROOT=Path(cfg['ustc_root'])
        relocate_run(module,cfg['result_root'])
        old_cache=module.CACHE_ROOT
        new_cache=Path(cfg['cache_root'])/'smgfnet_ustc'
        for key,value in list(vars(module).items()):
            if isinstance(value,Path) and value.is_relative_to(old_cache):
                setattr(module,key,new_cache/value.relative_to(old_cache))
    dsroot=cfg['cipher_root' if args.dataset=='cipher' else 'ustc_root']
    module.BASE.PCAP_DIR=dsroot
    module.CORE.PCAP_DIR=dsroot
    module.main()

def prepare_ibgc_raw(cfg,dataset):
    sys.path.insert(0,str(ROOT/'src/baselines/ibgc'))
    import baseline_v17_common as common
    class Logger:
        @staticmethod
        def log(message):
            print(message,flush=True)
    for spec in common.build_dataset_specs(dataset,ROOT,Path(cfg['result_root']),'IBGC'):
        common.prepare_raw_cache(spec,Path(cfg['cache_root'])/'baseline_raw',Logger())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('smgfnet','ablation',*ENTRIES),required=True)
    p.add_argument('--dataset',choices=('cipher','ustc','both'),default='cipher')
    p.add_argument('--mode',choices=('prepare','train','evaluate','aggregate','all'),default='all')
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--config')
    for opt in ('cipher-root','ustc-root','result-root','cache-root','etbert-root','smgf-seed42-checkpoint'):
        p.add_argument('--'+opt)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    cfg=settings(args)
    if args.model=='smgfnet':
        seeds=args.seeds.split(',')
        if not seeds or len(seeds)!=len(set(seeds)) or any(s not in ('42','43','44') for s in seeds):
            p.error('Seeds must be a unique subset of 42,43,44.')
        if args.dataset=='both' and args.seeds!='42,43,44':
            p.error('The combined archival run requires seeds 42,43,44.')
    if args.model in ('smgfnet','ablation') and args.mode!='all':
        p.error('Archived SMGF runners use mode=all; use analysis/roc/export_smgf_probabilities.py for inference-only export.')
    if args.model=='ablation' and (args.dataset!='cipher' or args.seeds!='42,43,44' and args.seeds!='42'):
        p.error('The matched ablation is CipherSpectrum seed 42 only.')
    if args.model=='ablation':
        args.seeds='42'
    if args.model not in ('smgfnet','ablation') and args.seeds!='42,43,44':
        p.error('The archived baseline runners use the formal three seeds 42,43,44.')
    if args.model=='smgfnet' and args.dataset=='cipher' and args.seeds!='42,43,44':
        p.error('The CipherSpectrum archival runner reuses seed42 and trains seeds43/44 as one formal run.')
    selected=('cipher','ustc') if args.dataset=='both' else (args.dataset,)
    if args.model=='ablation':
        selected=('cipher',)
    assert_preserved_sources()
    print(json.dumps(dict(model=args.model,dataset=args.dataset,mode=args.mode,seeds=args.seeds,
                          paths=cfg,dry_run=args.dry_run),indent=2),flush=True)
    if args.dry_run:
        return
    for ds in selected:
        if not cfg.get(f'{ds}_root') or not Path(cfg[f'{ds}_root']).is_dir():
            p.error(f'Set --{ds}-root or {ds}_root in your config to the original PCAP directory.')
    out=Path(cfg['result_root'])
    for protected in ('results','data','src','manifests','analysis','figures'):
        if out.is_relative_to(ROOT/protected):
            p.error('New training outputs must not overwrite the archived source/data/results.')
    if args.model in ('smgfnet','ablation') and 'cipher' in selected:
        checkpoint=cfg.get('smgf_seed42_checkpoint')
        if not checkpoint or not Path(checkpoint).is_file():
            p.error('CipherSpectrum archival orchestration requires --smgf-seed42-checkpoint; see docs/REPRODUCING.md.')
    if args.model=='smgfnet' and (args.dataset=='both' or args.dataset=='ustc' and ',' in args.seeds):
        seeds=args.seeds.split(',')
        if not seeds or len(seeds)!=len(set(seeds)) or any(s not in ('42','43','44') for s in seeds):
            p.error('Seeds must be a unique subset of 42,43,44.')
        for ds in selected:
            for seed in (['42,43,44'] if ds=='cipher' else seeds):
                command=[sys.executable,str(Path(__file__).resolve()),'--model','smgfnet','--dataset',ds,'--seeds',seed]
                for key,value in cfg.items():
                    command += ['--'+key.replace('_','-'),value]
                subprocess.run(command,cwd=ROOT,check=True)
        return
    env=environment(args,cfg)
    if args.model in ('smgfnet','ablation'):
        if args.model=='smgfnet' and args.dataset=='ustc' and args.seeds not in ('42','43','44'):
            p.error('USTC seeds must be 42,43,44.')
        native_smgf(args,cfg)
        return
    command=[sys.executable,str(ROOT/ENTRIES[args.model])]
    if args.model=='etbert':
        if not cfg.get('etbert_root') or not (Path(cfg['etbert_root'])/'fine-tuning/run_classifier.py').is_file():
            p.error('ET-BERT official source and external pretrained assets are required; see docs/THIRD_PARTY.md.')
    else:
        command += ['--mode',args.mode,'--datasets',args.dataset]
        if args.model in ('pean_nhp','ibgc'):
            command += ['--result-root',cfg['result_root']]
        if args.model=='ibgc':
            os.environ.update(env)
            if args.mode in ('prepare','all'):
                prepare_ibgc_raw(cfg,args.dataset)
            command += ['--work-root',str(ROOT),'--raw-cache-root',str(Path(cfg['cache_root'])/'baseline_raw')]
    subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':
    main()
