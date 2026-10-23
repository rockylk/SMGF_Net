"""Offline checks for archived sources, frozen splits, reports, and path adapters.

Uses only the Python standard library. Does not import ML dependencies, train,
connect to SSH/GitHub, or load a model checkpoint.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

def main():
    checks={}
    manifest=json.loads((ROOT/'manifests/source_manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path=ROOT/record['path']
        assert path.is_file(), record['path']
        assert sha(path)==record['sha256'], record['path']
        assert path.stat().st_size==record['bytes'], record['path']
        assert '云端' not in record['origin']
    checks['preserved_file_checksums']=len(manifest['files'])
    pyfiles=list(ROOT.rglob('*.py'))
    for p in pyfiles:
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    checks['python_sources_parsed']=len(pyfiles)
    split_report={}
    for ds,counts,classes in [('cipher',(29520,3280,8200),41),('ustc',(15304,1700,4252),12)]:
        obj=json.loads((ROOT/f'data/splits/{ds}.json').read_text(encoding='utf-8-sig'))
        assert obj.get('split_seed',42)==42
        seen_ids=set()
        seen_paths=set()
        for part,n in zip(('train','validation','test'),counts):
            rows=obj[part]
            assert len(rows)==n,(ds,part,len(rows))
            labels=set()
            for r in rows:
                assert r['sample_index'] not in seen_ids,(ds,'duplicate sample')
                assert r['relative_path'] not in seen_paths,(ds,'duplicate path')
                assert 0<=r['label']<classes
                assert not Path(r['relative_path']).is_absolute()
                assert '..' not in Path(r['relative_path']).parts
                seen_ids.add(r['sample_index'])
                seen_paths.add(r['relative_path'])
                labels.add(r['label'])
            assert labels==set(range(classes)),(ds,part,'missing class')
        split_report[ds]=dict(classes=classes,partition_counts=counts,unique_samples=len(seen_ids),status='PASS')
    checks['frozen_split_populations']=split_report

    ib=json.loads((ROOT/'results/raw/ibgc/run_manifest.json').read_text(encoding='utf-8-sig'))
    for name,expected in ib['code'].items():
        assert sha(ROOT/'src/baselines/ibgc'/name)==expected,name
    pean=json.loads((ROOT/'results/raw/pean_nhp/run_manifest.json').read_text(encoding='utf-8-sig'))
    pean_dir=ROOT/'src/baselines/pean_nhp'
    for name,expected in pean['upstream']['files_sha256'].items():
        assert sha(pean_dir/'official_sources/PEAN'/name)==expected,name
    for name,key in [('PEAN_NHP_final_paper_experiments_V1_100ep.py','runner_sha256'),
                     ('PEAN_NHP_final_data_V1_100ep.py','data_source_sha256'),
                     ('PEAN_NHP_V17_model.py','model_source_sha256'),
                     ('baseline_v17_common.py','common_sha256')]:
        assert sha(pean_dir/name)==pean[key],name
    et=json.loads((ROOT/'results/raw/etbert/run_manifest.json').read_text(encoding='utf-8-sig'))
    assert sha(ROOT/'src/baselines/etbert/ET-BERT_final_paper_experiments.py')==et['script_sha256']
    assert et['evaluation']['attack_base_seed']==20260923
    checks['baseline_manifest_source_locks']='PASS'

    summaries=load(ROOT/'scripts/summarize_results.py','archive_summary_validation')
    files=summaries.build()
    for name,text in files.items():
        assert (ROOT/name).read_text(encoding='utf-8')==text.replace('\r\n','\n'),name
    checks['generated_result_artifacts_verified']=len(files)

    launcher=load(ROOT/'scripts/run_experiment.py','archive_launcher_validation')
    old=ROOT/'__test_old_result__'
    fake=SimpleNamespace(RUN_DIR=old,RESULT_ROOT=ROOT/'old',CHECKPOINT_DIR=old/'checkpoints',
                         CONFIG_FILE=old/'run_config.json',BASE_SOURCE=ROOT/'source.py')
    launcher.relocate_run(fake,ROOT/'outputs/test_result')
    assert fake.RUN_DIR==ROOT/'outputs/test_result'/old.name
    assert fake.CHECKPOINT_DIR==fake.RUN_DIR/'checkpoints'
    assert fake.CONFIG_FILE==fake.RUN_DIR/'run_config.json'
    assert fake.BASE_SOURCE==ROOT/'source.py'
    assert launcher.resolve('data/splits/cipher.json')==ROOT/'data/splits/cipher.json'
    checks['portable_path_adapter']='PASS'
    for model in ('smgfnet','ablation','ustc_cnn','etbert','pean_nhp','ibgc'):
        command=[sys.executable,str(ROOT/'scripts/run_experiment.py'),'--model',model,'--dry-run']
        p=subprocess.run(command,cwd=ROOT/'docs',text=True,encoding='utf-8',capture_output=True)
        assert p.returncode==0,(model,p.stderr)
        assert json.loads(p.stdout)['dry_run'] is True,model
    checks['offline_cli_dry_runs']=6
    failed=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),'--model','smgfnet',
                           '--cipher-root',str(ROOT/'data')],cwd=ROOT/'docs',
                          text=True,encoding='utf-8',capture_output=True)
    assert failed.returncode!=0 and 'requires --smgf-seed42-checkpoint' in failed.stderr
    checks['missing_original_checkpoint_fails_before_ml_import']='PASS'
    modules={n.split('.')[0] for n in sys.modules}
    assert not modules.intersection({'torch','torch_geometric','scapy','matplotlib','numpy'})
    checks['no_ml_import_or_training']='PASS'

    missing=[]
    docs=[ROOT/'README.md',ROOT/'data/README.md']+list((ROOT/'docs').glob('*.md'))+list((ROOT/'results').glob('*.md'))
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            if not (doc.parent/target.split('#')[0]).exists():
                missing.append((doc.relative_to(ROOT).as_posix(),target))
    assert not missing,missing
    checks['documentation_links']='PASS'
    forbidden=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and (p.suffix.lower() in ('.pcap','.pcapng','.pt','.pth','.bin','.pem','.key','.zip','.log')
                            or p.name.lower().startswith('.env') or 'ssh_dependencies' in p.parts):
            forbidden.append(p.relative_to(ROOT).as_posix())
    assert not forbidden,forbidden
    checks['no_raw_traffic_weights_credentials_or_training_logs']='PASS'
    findings=[]
    secret_patterns=[r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',
                     r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',r'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
                     r'(?i)(?:password|passwd|api_key|access_token)\s*[:=]\s*[\"\x27][^\"\x27\s]{8,}[\"\x27]']
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in ('.py','.json','.md','.sh','.txt') or p.name==Path(__file__).name:
            continue
        text=p.read_text(encoding='utf-8-sig')
        if any(re.search(pattern,text) for pattern in secret_patterns):
            findings.append(p.relative_to(ROOT).as_posix())
    assert not findings,('Potential hardcoded credential, inspect locally',findings)
    checks['hardcoded_credential_pattern_scan']='PASS'
    attrs=(ROOT/'.gitattributes').read_text()
    assert attrs.index('*.md')<attrs.index('src/** -text')
    checks['frozen_git_newline_policy']='PASS'

    report=dict(status='PASS',validation_scope='Offline packaging and archived-result checks only',
                checks=checks,not_run=['GPU training','checkpoint inference','GPU profiling','raw PCAP regeneration',
                                     'GitHub publication','upstream downloads'])
    (ROOT/'manifests/validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
