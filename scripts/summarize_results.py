"""Regenerate paper result tables from archived observations, without ML dependencies."""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw'
MODELS = ('USTC-CNN', 'ET-BERT', 'PEAN-NHP', 'IBGC', 'SMGF-Net')
DATASETS = ('CipherSpectrum', 'USTC-TFC2016')
SEEDS = (42, 43, 44)
CONDITIONS = ('clean', 'mixed_1.0', 'adaptive_1.0')
METRICS = ('accuracy', 'macro_f1', 'asr', 'relative_drop', 'ece', 'auroc')

def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def number(row, *keys):
    for k in keys:
        if k in row and row[k] != '':
            x = float(row[k])
            if not math.isfinite(x):
                raise ValueError(f'Non-finite metric {k}')
            return x
    raise KeyError(keys)

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def csv_text(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()

def normalized_observations():
    rows = []
    for model, folder, filename in [
        ('USTC-CNN','ustc_cnn','USTC_CNN_per_seed_primary.csv'),
        ('ET-BERT','etbert','ETBERT_per_seed_primary.csv'),
        ('PEAN-NHP','pean_nhp','PEAN_NHP_per_seed_primary.csv'),
        ('SMGF-Net','smgfnet_cipher','paper_ready_primary_per_seed.csv'),
        ('SMGF-Net','smgfnet_ustc','paper_ready_primary_per_seed.csv')]:
        for r in read_csv(RAW/folder/filename):
            dataset = r.get('dataset') or ('CipherSpectrum' if folder.endswith('cipher') else 'USTC-TFC2016')
            for cond in CONDITIONS:
                clean = cond == 'clean'
                attack = cond.split('_')[0]
                if clean:
                    acc = number(r,'clean_acc')
                    f1 = number(r,'clean_macro_f1','clean_f1')
                    ece = number(r,'clean_ece')
                    asr = drop = ''
                else:
                    acc = number(r,f'{attack}_robust_acc',f'{attack}1_robust_acc',f'{attack}1_acc')
                    f1 = number(r,f'{attack}_robust_f1',f'{attack}1_robust_f1',f'{attack}1_f1')
                    ece = number(r,f'{attack}_ece',f'{attack}1_ece')
                    asr = number(r,f'{attack}_asr',f'{attack}1_asr')
                    drop = number(r,f'{attack}_rel_drop',f'{attack}1_rel_drop')
                rows.append(dict(model=model,dataset=dataset,seed=int(r['seed']),condition=cond,
                                 accuracy=acc,macro_f1=f1,asr=asr,relative_drop=drop,ece=ece,auroc=''))
    ib = defaultdict(dict)
    for r in read_csv(RAW/'ibgc/per_seed.csv'):
        ib[(r['dataset'],int(r['seed']),r['condition'])][r['metric']] = float(r['value'])
    for (ds,seed,cond),r in ib.items():
        if cond not in CONDITIONS:
            continue
        rows.append(dict(model='IBGC',dataset=ds,seed=seed,condition=cond,
                         accuracy=number(r,'acc','robust_acc'),macro_f1=number(r,'macro_f1','robust_f1'),
                         asr='' if cond=='clean' else number(r,'asr'),
                         relative_drop='' if cond=='clean' else number(r,'rel_drop'),
                         ece=number(r,'ece'),auroc=''))
    index = {(r['model'],r['dataset'],r['seed'],r['condition']):r for r in rows}
    if len(rows) != 90 or len(index) != 90:
        raise AssertionError('Expected 5 models x 2 datasets x 3 seeds x 3 conditions')
    seen = set()
    for r in read_csv(ROOT/'analysis/roc/per_seed_auroc_and_provenance.csv'):
        key = (r['model'],r['dataset'],int(r['seed']),r['condition'])
        assert key not in seen, key
        seen.add(key)
        dest = index[key]
        assert abs(dest['accuracy']-float(r['accuracy_pct'])) < 1e-8, key
        assert int(r['n']) == (8200 if r['dataset']=='CipherSpectrum' else 4252)
        assert int(r['classes']) == (41 if r['dataset']=='CipherSpectrum' else 12)
        dest['auroc'] = 100*number(r,'macro_ovr_auroc')
    assert len(seen)==90
    for r in rows:
        for m in METRICS:
            if r[m] != '':
                assert 0 <= r[m] <= 100, (r,m)
    return sorted(rows,key=lambda r:(DATASETS.index(r['dataset']),MODELS.index(r['model']),r['seed'],CONDITIONS.index(r['condition'])))

def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['model'],r['dataset'],r['condition'])].append(r)
    output=[]
    for (model,ds,cond), group in grouped.items():
        assert sorted(r['seed'] for r in group)==list(SEEDS)
        record=dict(model=model,dataset=ds,condition=cond,n_models=3)
        for metric in METRICS:
            vals=[r[metric] for r in group if r[metric]!='']
            record[metric+'_mean']=statistics.mean(vals) if vals else ''
            record[metric+'_sample_sd']=statistics.stdev(vals) if vals else ''
        output.append(record)
    idx={(r['model'],r['dataset'],r['condition']):r for r in output}
    for r in read_csv(ROOT/'analysis/roc/auroc_mean_sample_sd.csv'):
        dest=idx[(r['model'],r['dataset'],r['condition'])]
        assert abs(dest['auroc_mean']-100*float(r['mean_auroc']))<1e-8
        assert abs(dest['auroc_sample_sd']-100*float(r['sample_sd_auroc']))<1e-8
    return output,idx

def fmt(r, metric):
    if r[metric+'_mean']=='':
        return 'N/A'
    return f"{r[metric+'_mean']:.2f} ± {r[metric+'_sample_sd']:.2f}"

def verify_archived_aggregates(idx):
    """Independently compare regenerated estimates with each original summary."""
    count=0
    def compare(model,ds,cond,metric,mean,sd):
        nonlocal count
        r=idx[(model,ds,cond)]
        assert abs(r[metric+'_mean']-float(mean))<1e-8,(model,ds,cond,metric,'mean')
        assert abs(r[metric+'_sample_sd']-float(sd))<1e-8,(model,ds,cond,metric,'sd')
        count+=2
    def parse_key(key):
        for prefix,cond in [('clean_','clean'),('mixed1_','mixed_1.0'),('mixed_','mixed_1.0'),
                            ('adaptive1_','adaptive_1.0'),('adaptive_','adaptive_1.0')]:
            if key.startswith(prefix):
                m=key[len(prefix):]
                mapped={'acc':'accuracy','robust_acc':'accuracy','f1':'macro_f1','macro_f1':'macro_f1',
                        'robust_f1':'macro_f1','asr':'asr','rel_drop':'relative_drop','ece':'ece',
                        'auroc':'auroc','macro_auroc':'auroc'}.get(m)
                return cond,mapped
        return None,None
    for model,folder,filename in [('USTC-CNN','ustc_cnn','USTC_CNN_aggregate_primary.csv'),
                                  ('ET-BERT','etbert','ETBERT_aggregate_primary.csv')]:
        for r in read_csv(RAW/folder/filename):
            for key in r:
                if key.endswith('_mean'):
                    stem=key[:-5]
                    cond,metric=parse_key(stem)
                    if metric:
                        compare(model,r['dataset'],cond,metric,r[key],r[stem+'_sample_sd'])
    for model,folder,filename,default_ds in [
        ('PEAN-NHP','pean_nhp','PEAN_NHP_aggregate_primary.csv',None),
        ('SMGF-Net','smgfnet_ustc','paper_ready_aggregate_metrics.csv','USTC-TFC2016')]:
        for r in read_csv(RAW/folder/filename):
            cond,metric=parse_key(r['metric'])
            if metric:
                compare(model,r.get('dataset',default_ds),cond,metric,r['mean'],r['sample_sd'])
    for r in read_csv(RAW/'ibgc/aggregate.csv'):
        cond=r['condition']
        metric={'acc':'accuracy','robust_acc':'accuracy','macro_f1':'macro_f1','robust_f1':'macro_f1',
                'asr':'asr','rel_drop':'relative_drop','ece':'ece','macro_auroc':'auroc'}.get(r['metric'])
        if cond in CONDITIONS and metric:
            compare('IBGC',r['dataset'],cond,metric,r['mean'],r['sample_sd'])
    for r in read_csv(RAW/'smgfnet_cipher/paper_ready_aggregate_metrics.csv'):
        cond=r['condition']
        if cond!='clean':
            if cond not in ('mixed','adaptive') or float(r['strength'])!=1:
                continue
            cond+='_1.0'
        for stem,metric in [('acc','accuracy'),('f1','macro_f1'),('asr','asr'),('rel_drop','relative_drop'),('ece','ece')]:
            if r[stem+'_mean']!='':
                compare('SMGF-Net','CipherSpectrum',cond,metric,r[stem+'_mean'],r[stem+'_sd'])
    return count

def sensitivity(observations):
    db={}
    def add(model,seed,attack,strength,value):
        key=(model,attack,float(strength),int(seed))
        assert key not in db, key
        db[key]=dict(model=model,attack=attack,strength=float(strength),seed=int(seed),accuracy=float(value))
    for r in observations:
        if r['dataset']!='CipherSpectrum' or r['model']=='USTC-CNN':
            continue
        if r['condition']=='clean':
            for a in ('mixed','adaptive'):
                add(r['model'],r['seed'],a,0,r['accuracy'])
        else:
            add(r['model'],r['seed'],r['condition'].split('_')[0],1,r['accuracy'])
    for r in read_csv(RAW/'sensitivity/Figure6_CrossModel_45_verified.csv'):
        add(r['model'],r['seed'],r['attack'],r['strength'],r['robust_acc'])
    source=json.loads((RAW/'smgfnet_cipher/per_seed_results.json').read_text(encoding='utf-8-sig'))
    for r in source['test_results']:
        for k,v in r['attacks'].items():
            attack,strength=k.split('@')
            if attack in ('mixed','adaptive') and float(strength)!=1:
                add('SMGF-Net',r['seed'],attack,strength,v['robust_acc'])
    assert len(db)==108
    grouped=defaultdict(list)
    for key,r in db.items():
        grouped[key[:3]].append(r)
    summary=[]
    for (m,a,s),group in sorted(grouped.items()):
        assert sorted(r['seed'] for r in group)==list(SEEDS)
        vals=[r['accuracy'] for r in group]
        summary.append(dict(model=m,attack=a,strength=s,mean=statistics.mean(vals),sample_sd=statistics.stdev(vals)))
    return list(db.values()),summary

def build():
    obs=normalized_observations()
    agg,idx=aggregate(obs)
    matched=verify_archived_aggregates(idx)
    def result_table(ds,cond):
        metrics=('accuracy','macro_f1','ece') if cond=='clean' else ('accuracy','macro_f1','asr','relative_drop','ece')
        headers=['Model']+(['Accuracy (%)','Macro-F1 (%)','ECE (%)'] if cond=='clean' else
                          ['Robust accuracy (%)','Robust Macro-F1 (%)','ASR (%)','Relative drop (%)','ECE (%)'])
        return table(headers,[[m]+[fmt(idx[(m,ds,cond)],v) for v in metrics] for m in MODELS])
    lines=['# Experimental results','',
      'Generated from the archived final per-seed observations. All performance metrics are percentages unless noted. Values are mean ± sample SD over independently trained models (seeds 42, 43, 44), with ddof=1. No new training or GPU measurements are performed by this script.','',
      '## Evaluation population','',
      table(['Dataset','Classes','Train','Validation','Test'],[
          ['CipherSpectrum',41,29520,3280,8200],['USTC-TFC2016 archived subset',12,15304,1700,4252]]),'',
      'USTC refers to the archived 12-class, 21,256-flow subset, not the complete original corpus.','']
    for ds in DATASETS:
        lines += [f'## {ds}','', '### Clean','',result_table(ds,'clean'),'']
        for c,label in [('mixed_1.0','Mixed@1.0'),('adaptive_1.0','Adaptive@1.0')]:
            lines += [f'### {label}','',result_table(ds,c),'']
    lines += ['## Macro one-versus-rest AUROC','',
      table(['Dataset','Model','Clean (%)','Mixed@1.0 (%)','Adaptive@1.0 (%)'],
            [[ds,m]+[fmt(idx[(m,ds,c)],'auroc') for c in CONDITIONS] for ds in DATASETS for m in MODELS]),'',
      'For each seed, compute an OvR AUC for every class and average all classes equally; then summarize the three seed-level AUCs. Seeds are not pooled, probabilities are not averaged between models, and displayed interpolated-curve areas do not replace original AUROC. ROC bands show cross-model sample SD, not 95% confidence intervals.','']
    ab=read_csv(RAW/'ablation/PAPER_READY_CIPHER_MATCHED_ABLATION.csv')
    labels={'FULL':'Full model','STRICT_NO_GATE':'w/o Learned Gating','CLEAN_ONLY':'w/o Robust Training','NO_GATE_CLEAN_ONLY':'w/o Learned Gating & Robust Training'}
    at=table(['Setting','Clean Acc (%)','Mixed RobAcc (%)','Mixed RelDrop (%)','Adaptive RobAcc (%)','Adaptive RelDrop (%)'],
      [[labels[r['Variant']]]+[f'{float(r[k]):.2f}' for k in ('CleanAcc','MixedRobAcc','MixedRelDrop','AdaptiveRobAcc','AdaptiveRelDrop')] for r in ab])
    lines += ['## Matched component ablation (Table VII)','',at,'',
      'CipherSpectrum; fixed training seed 42; identical split, initialization seed, optimization schedule, clean-validation checkpoint criterion, and deterministic attacks. Full reuses the frozen main-experiment seed-42 checkpoint. These are single-seed diagnostic comparisons, not three-model means.','']
    abi={r['Variant']:r for r in ab}
    effects={}
    for key,label in [('MixedRobAcc','Mixed'),('AdaptiveRobAcc','Adaptive')]:
        full,ng,nt,both=[float(abi[x][key]) for x in ('FULL','STRICT_NO_GATE','CLEAN_ONLY','NO_GATE_CLEAN_ONLY')]
        effects[label]=dict(gate_with_training_pp=full-ng,gate_without_training_pp=nt-both,
                           training_with_gate_pp=full-nt,training_without_gate_pp=ng-both,
                           interaction_pp=full-ng-nt+both)
    contrast_labels={'gate_with_training_pp':'Gate effect with robust training',
                     'gate_without_training_pp':'Gate effect without robust training',
                     'training_with_gate_pp':'Training effect with learned Gate',
                     'training_without_gate_pp':'Training effect with identity Gate',
                     'interaction_pp':'Gate × training interaction'}
    lines += [table(['Contrast (percentage points)','Mixed','Adaptive'],
      [[contrast_labels[k]]+[f'{effects[a][k]:.2f}' for a in ('Mixed','Adaptive')] for k in next(iter(effects.values()))]),'',
      'Interactions are subadditive in this fixed-seed comparison; they are not evidence of statistical significance or synergy.','']
    overhead=read_csv(RAW/'overhead/TableVIII_V17_status.csv')
    ot=table(['Model','Parameters (M)','Profiled FLOPs subtotal (M)','GPU forward latency (ms/flow)'],
      [[r['Model'],f"{float(r['Parameters (M)']):.3f}",
        f"{float(r['Supported-Op FLOPs Subtotal (M)'].split()[0]):,.3f}",
        ' ± '.join(f'{float(v.strip()):.3f}' for v in r['Inference Latency (ms/flow)'].split('?'))] for r in overhead])
    lines += ['## Computational overhead (Table VIII)','',ot,'',
      'CipherSpectrum, batch size 1, eight held-out flows per seed, 30 warm-up iterations and 100 timed forward passes per flow. Latencies are averaged within each seed and summarized across seeds. These are archived model-forward measurements, excluding PCAP parsing, preprocessing, and transfer overhead. FLOPs are the profiler-supported operator subtotal, not a complete model-wide count; the original measurement status remains MEASURED_FLOPS_UNVERIFIED. The source CSV lost the ± character as “?”; only its display is restored here.','']
    gate=read_csv(RAW/'gate_audit/Figure10_GateAudit_strength_summary.csv')
    for r in gate:
        vals=[float(r[f'seed{s}_mean_abs_gate_shift']) for s in SEEDS]
        assert abs(statistics.mean(vals)-float(r['mean_abs_gate_shift_mean']))<1e-12
        assert abs(statistics.stdev(vals)-float(r['mean_abs_gate_shift_sd']))<1e-12
    gt=table(['Attack','Strength','Mean absolute paired Gate shift ± sample SD'],
      [[r['attack'],r['strength'],f"{float(r['mean_abs_gate_shift_mean']):.5f} ± {float(r['mean_abs_gate_shift_sd']):.5f}"] for r in gate if float(r['strength'])>0])
    lines += ['## Channel-level Gate audit','',gt,'',
      'The audit uses a deterministic class-balanced subset of 1,000 CipherSpectrum test flows and the three frozen models. Channel shifts are bidirectional. Their magnitude describes modulation, not a causal measure of classification benefit. Historical archive filenames call this Figure10; the current manuscript uses Fig. 7.','',
      '## Strength sensitivity','',
      'See [SENSITIVITY.md](SENSITIVITY.md) for all measured Mixed and Adaptive strength points for SMGF-Net, ET-BERT, PEAN-NHP, and IBGC on CipherSpectrum. Intermediate baseline values are the 45 verified GPU observations; no missing values are interpolated.','',
      '## Reproduction and source records','',
      '- [Normalized per-seed observations](tables/per_seed_metrics.csv) and [aggregate metrics](tables/aggregate_metrics.csv).',
      '- [Original result snapshots](raw/) and [source checksum manifest](../manifests/source_manifest.json).',
      '- [Protocol and model adaptations](../docs/PROTOCOL.md).',
      '- [Reproduction instructions](../docs/REPRODUCING.md).','']
    sens,sens_summary=sensitivity(obs)
    sl=['# CipherSpectrum strength sensitivity','',
        'Accuracy (%), mean ± sample SD across seeds 42, 43, 44. Strength 0 is the clean reference. No fitted or interpolated observations.','']
    si={(r['model'],r['attack'],r['strength']):r for r in sens_summary}
    for a,strengths in [('mixed',(0,.3,.6,.9,1)),('adaptive',(0,.4,.7,1))]:
        sl += [f'## {a.title()}','',table(['Strength','SMGF-Net','PEAN-NHP','IBGC','ET-BERT'],
          [[s]+[f"{si[(m,a,s)]['mean']:.2f} ± {si[(m,a,s)]['sample_sd']:.2f}" for m in ('SMGF-Net','PEAN-NHP','IBGC','ET-BERT')] for s in strengths]),'']
    files={'results/RESULTS.md':'\n'.join(lines),'results/SENSITIVITY.md':'\n'.join(sl),
           'results/tables/per_seed_metrics.csv':csv_text(obs),
           'results/tables/aggregate_metrics.csv':csv_text(agg),
           'results/tables/sensitivity_per_seed.csv':csv_text(sens),
           'results/tables/sensitivity_aggregate.csv':csv_text(sens_summary),
           'results/tables/ablation_effects.json':json.dumps(effects,indent=2)+'\n'}
    files['results/tables/verification.json']=json.dumps(dict(
        primary_per_seed_records=90,complete_three_seed_groups=30,
        original_aggregate_values_verified=matched,probability_accuracy_checks=90,
        roc_summary_groups_verified=30,gate_summary_groups_verified=len(gate),
        sensitivity_per_seed_records=108,statistical_unit='independently trained model',
        sample_sd_ddof=1,pooled_seeds=False,averaged_probabilities=False,
        status='PASS'),indent=2)+'\n'
    for model in MODELS:
        name={'USTC-CNN':'ustc_cnn','ET-BERT':'etbert','PEAN-NHP':'pean_nhp','IBGC':'ibgc','SMGF-Net':'smgfnet'}[model]
        ml=[f'# {model}: archived results','', 'Percentages; mean ± sample SD over seeds 42, 43, 44.','',
            table(['Dataset','Condition','Accuracy','Macro-F1','ASR','Relative drop','ECE','Macro OvR AUROC'],
              [[ds,c]+[fmt(idx[(model,ds,c)],k) for k in METRICS] for ds in DATASETS for c in CONDITIONS]),'',
            'See [RESULTS.md](RESULTS.md) and [PROTOCOL.md](../docs/PROTOCOL.md) for population, aggregation, and input adaptations.','']
        files[f'results/{name}.md']='\n'.join(ml)
    return files

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true',help='Check generated files without rewriting them')
    args=p.parse_args()
    files=build()
    for name,text in files.items():
        dest=ROOT/name
        if args.check:
            assert dest.read_text(encoding='utf-8')==text.replace('\r\n','\n'), f'Stale generated file: {name}'
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(text,encoding='utf-8',newline='')
    print(json.dumps(dict(generated_files=len(files),primary_per_seed_records=90,
                         primary_three_seed_groups=30,sensitivity_per_seed_records=108,
                         status='PASS'),indent=2))

if __name__=='__main__':
    main()
