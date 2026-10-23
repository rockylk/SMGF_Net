#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONE command server-side Section 5.6 measurement suite, frozen V17.

Uses /home/admin1/lkfiles for ORIGINAL sources and /home/admin1/lkfiles/RESULT
for ORIGINAL checkpoint/manifests/clean test caches. Never trains, never writes
to any official snapshots. Runs in the EXISTING GPU Python environment.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from inventory import RESULT,ROOT,SEEDS,MODEL_DIRS,sha256,checkpoint_path,source_path

HERE=Path(__file__).resolve().parent
IDX=(0,100,500,1000,1500,2000,4000,6000)
ORDER=("USTC-CNN","ET-BERT","PEAN-NHP","IBGC","SMGF-Net")

def emit(*msg):
    print("[Section56]",*msg,flush=True)

def pjson(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    tmp.replace(path)

def canonical():
    path=RESULT/"SMGF_V17_CIPHERSPECTRUM_FINAL_3SEED_20260923_135542"/"split_manifest.json"
    if not path.is_file():raise FileNotFoundError("Original frozen SMGF split manifest missing: "+str(path))
    data=json.loads(path.read_text(encoding="utf-8"))
    assert data["dataset_name"]=="CipherSpectrum"
    assert data["dataset_manifest_sha256"]=="821bee04771f0c8dce3680a4301164d8ab48e03ec0d1ed57c3930015ad200a61"
    assert len(data["test"])==8200
    return path,data["test"]

def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible; this is GPU-only benchmark and will not substitute CPU results")
    torch.cuda.set_device(0)
    emit("GPU",torch.cuda.get_device_name(0),"torch",torch.__version__,
         "CUDA",torch.version.cuda)
    return torch

def run_child(cmd,log,env,desc):
    log.parent.mkdir(parents=True,exist_ok=True)
    emit("EXEC",desc,"log",str(log))
    with log.open("w",encoding="utf-8") as f:
        f.write("COMMAND: "+json.dumps([str(x) for x in cmd])+"\n")
        f.flush()
        r=subprocess.run([str(x) for x in cmd],env=env,cwd=str(ROOT),stdout=f,
                         stderr=subprocess.STDOUT,check=False)
    if r.returncode:
        tail="\n".join(log.read_text(encoding="utf-8",errors="replace").splitlines()[-17:])
        raise RuntimeError(f"{desc}: process exit {r.returncode}, log={log}\n{tail}")
    return log

def candidate_raw_cache():
    formal=RESULT/"IBGC_V17_V1_100EP"/"cipher"
    meta=json.loads((formal/"preparation.json").read_text(encoding="utf-8"))
    target=meta["identity"]["raw_cache_sha256"]["test"]
    provided=os.environ.get("S56_IBGC_RAW_TEST")
    found=[]
    if provided:found.append(Path(provided))
    found.extend([
        formal/"prepared"/"test_clean.npz",
        formal/"raw_cache"/"test.npz",formal/"raw_cache"/"test_clean.npz",
        formal/"cache"/"test.npz",formal/"cache"/"test_clean.npz",
        RESULT/"IBGC_V17_V1_100EP"/"raw_cache"/"test.npz",
        ROOT/"RESULT"/"IBGC_V17_V1_100EP"/"cipher"/"prepared"/"test_clean.npz",
    ])
    for root in (formal,RESULT):
        # Path discovery is metadata-only until a candidate path resembles raw test cache;
        # upper bound avoids exhaustive reading of large unrelated NPZ files.
        limit=0
        for pattern in ("**/*raw*test*.npz","**/test_raw*.npz",
                        "**/raw_cache/test*.npz"):
            for f in root.glob(pattern):
                found.append(f);limit+=1
                if limit>=100:break
            if limit>=100:break
    tested=set()
    for p in found:
        if p in tested or not p.is_file():continue
        tested.add(p)
        try:
            if sha256(p)==target:
                emit("IBGC verified raw test cache",p)
                return p
        except OSError:
            continue
    raise FileNotFoundError(
        "IBGC frozen raw_test_cache not found with SHA="+target+
        ". Set S56_IBGC_RAW_TEST=/absolute/original/npz and rerun. "
        "Will not use prepared graph cache or synthetic graph as a substitute.")

def valid_graph_inputs(model,dir):
    for idx in IDX:
        path=dir/f"{model.replace('-','_')}_cipher_sample{idx}.pt"
        meta=path.with_suffix(".provenance.json")
        if not path.is_file() or not meta.is_file():return False
        try:
            obj=json.loads(meta.read_text(encoding="utf-8"))
            if obj.get("model")!=model or obj.get("index")!=idx or obj.get("output_sha256")!=sha256(path):
                return False
        except (OSError,ValueError):return False
    return True

def graph_inputs(model,dirs,env):
    d=dirs["graph"]
    if valid_graph_inputs(model,d):emit("REUSE authenticated graph inputs",model);return
    cmd=[sys.executable,str(HERE/"export_graph_input.py"),"--model",model,
         "--indices",",".join(map(str,IDX)),"--output-dir",str(d)]
    if model=="SMGF-Net":
        cmd+=["--pcap-root",os.environ.get("S56_PCAP_ROOT","/home/admin1/dataset/aes-128-gcm/aes-128-gcm")]
    else:
        cmd+=["--raw-test-cache",str(candidate_raw_cache())]
    run_child(cmd,dirs["log"]/f"prepare_{model.replace('-','_')}.log",env,"graph export "+model)
    if not valid_graph_inputs(model,d):raise RuntimeError("Graph export incomplete: "+model)

def run_pytorch(model,dirs,env):
    for seed in SEEDS:
        for idx in IDX:
            name=f"{model.replace('-','_')}_cipher_seed{seed}_sample{idx}.json"
            dest=dirs["meas"]/name
            if dest.is_file():
                # Reuse only complete and still-verifiable measurements. Old
                # version trusted filename/three fields; this was insufficient.
                obj=json.loads(dest.read_text(encoding="utf-8"))
                expect_ckpt=checkpoint_path(model,seed)
                valid=(obj.get("schema")=="section56-v17-modelonly-v1"
                    and obj.get("model")==model and obj.get("seed")==seed
                    and obj.get("sample_index")==idx and obj.get("device")=="cuda"
                    and obj.get("checkpoint_sha256") == sha256(expect_ckpt)
                    and obj.get("checkpoint") == str(expect_ckpt)
                    and float(obj.get("latency",{}).get("median_ms",0))>0
                    and obj.get("flops",{}).get("coverage")=="UNVERIFIED_SUBTOTAL_NOT_TOTAL")
                for field in ("source","input_path"):
                    path=Path(obj.get(field,""))
                    hashfield="source_sha256" if field=="source" else "input_sha256"
                    valid=valid and path.is_file() and obj.get(hashfield)==sha256(path)
                if valid:
                    emit("REUSE_VERIFIED",model,seed,idx);continue
                raise RuntimeError("Existing profile is incomplete or unverified; kept unchanged for audit: "+str(dest))
            cmd=[sys.executable,str(HERE/"profile_one.py"),
                 "--model",model,"--dataset","cipher","--seed",str(seed),
                 "--sample-index",str(idx),"--warmup",env["S56_WARMUP"],
                 "--repeat",env["S56_REPEATS"],"--device","cuda","--out",str(dirs["meas"])]
            if model in ("SMGF-Net","IBGC"):
                cmd+=["--input-path",str(dirs["graph"]/f"{model.replace('-','_')}_cipher_sample{idx}.pt")]
            run_child(cmd,dirs["log"]/f"{model.replace('-','_')}_s{seed}_i{idx}.log",
                      env,f"{model} seed{seed} sample{idx}")

def complete_and_attested(model, dirs, env, split, gpu_name, torch_version):
    """Reuse all 24 frozen measurements ONLY if original evidence still verifies.

    Particularly important for ET-BERT: official inference is expensive and
    must not run again just because previous measurements exist.
    This function validates samples, hashes, device and profiling protocol.
    """
    digest_cache={}
    def digest(path):
        key=str(Path(path).resolve())
        if key not in digest_cache:
            file=Path(key)
            digest_cache[key]=sha256(file) if file.is_file() else None
        return digest_cache[key]
    expected_gpu=str(gpu_name)
    for seed in SEEDS:
        ckpt=checkpoint_path(model,seed)
        expected_hash=digest(ckpt)
        if not expected_hash:
            return False
        for idx in IDX:
            file=dirs["meas"]/f"{model.replace('-','_')}_cipher_seed{seed}_sample{idx}.json"
            if not file.is_file():
                return False
            try:
                r=json.loads(file.read_text(encoding="utf-8"))
                if not (
                    r.get("schema")=="section56-v17-modelonly-v1"
                    and r.get("model")==model and r.get("dataset")=="cipher"
                    and r.get("seed")==seed and r.get("sample_index")==idx
                    and r.get("sample_id")==int(split[idx]["sample_index"])
                    and r.get("device")=="cuda"
                    and r.get("gpu_name")==expected_gpu
                    and str(r.get("torch_version"))==str(torch_version)
                    and r.get("batch_size")==1
                    and r.get("warmup")==int(env["S56_WARMUP"])
                    and r.get("repeats")==int(env["S56_REPEATS"])
                    and r.get("checkpoint_sha256")==expected_hash
                    and digest(r.get("checkpoint",""))==expected_hash
                    and float(r["latency"]["median_ms"])>0
                    and r["flops"]["coverage"]=="UNVERIFIED_SUBTOTAL_NOT_TOTAL"
                ):
                    return False
                for field,hash_field in (("source","source_sha256"),
                                          ("input_path","input_sha256")):
                    if digest(r.get(field,"")) != r.get(hash_field):
                        return False
            except (OSError, ValueError, TypeError, KeyError):
                return False
    return True

def etbert_official(seed,dirs,env,split):
    import torch
    official=Path(os.environ.get("S56_ETBERT_REPO",str(ROOT/"ET-BERT")))
    inference=official/"inference"/"run_classifier_infer.py"
    config=official/"models"/"bert"/"base_config.json"
    if not config.is_file():config=official/"bert_base_config.json"
    vocab=official/"models"/"encryptd_vocab.txt"
    if not vocab.is_file():vocab=official/"models"/"encrypted_vocab.txt"
    for f in (inference,config,vocab):
        if not f.is_file():raise FileNotFoundError("Official ET-BERT dependency absent: "+str(f))
    run=RESULT/"ETBERT_FINAL_PAPER_BASELINE_V1"
    ckpt=run/"cipher"/"models"/f"seed{seed}"/"finetuned.bin"
    summary=ckpt.parent/"train_summary.json"
    prep=run/"cipher"/"dataset_preparation.json"
    tsv=run/"cipher"/"datasets"/"test_clean.tsv"
    for f in (ckpt,summary,prep,tsv):
        if not f.is_file():raise FileNotFoundError("Frozen ET-BERT artifact absent: "+str(f))
    sd=json.loads(summary.read_text(encoding="utf-8"))
    if sd.get("seed")!=seed or sd.get("checkpoint_sha256")!=sha256(ckpt):
        raise RuntimeError("Official ET-BERT checkpoint/seed provenance mismatch")
    pre=json.loads(prep.read_text(encoding="utf-8"))
    if pre.get("split_sha256")!="01e3b572e8514a238b3733e224161acd1112b79b56f038635be9339b23cfad6b" or \
       pre.get("tsv_sha256",{}).get("test_clean")!=sha256(tsv):
        raise RuntimeError("Official ET-BERT test TSV manifest/SHA differs from frozen run")
    # NO token regeneration: slice original exact TSV from server with row indices.
    with tsv.open("r",encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f,delimiter="\t")
        if tuple(reader.fieldnames or ())!=("label","text_a"):
            raise ValueError("ET-BERT frozen TSV schema changed")
        chosen=[(pos,row) for pos,row in enumerate(reader) if pos in IDX]
    if len(chosen)!=len(IDX):raise RuntimeError("ET-BERT frozen TSV missing requested samples")
    for idx,row in chosen:
        if int(row["label"])!=int(split[idx]["label"]):
            raise RuntimeError(f"ET-BERT versus SMGF frozen test label mismatch index={idx}")
    subset=dirs["tmp"]/"ETBERT_cipher_authentic_8flows.tsv"
    subset.parent.mkdir(parents=True,exist_ok=True)
    with subset.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["label","text_a"],delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for _,row in chosen)
    raw=dirs["tmp"]/f"etbert_hook_seed{seed}.jsonl"
    if raw.exists():raw.unlink() # only our own temp hook output
    prediction=dirs["tmp"]/f"etbert_prediction_seed{seed}.tsv"
    child=env.copy()
    child["PYTHONPATH"]=os.pathsep.join([
        str(HERE/"etbert_hook"),str(official),str(official/"fine-tuning"),
        child.get("PYTHONPATH","")])
    child["MKL_THREADING_LAYER"]="GNU"
    child.pop("MKL_SERVICE_FORCE_INTEL",None)
    child["S56_ET_BERT_HOOK_OUT"]=str(raw)
    child["S56_EXPECTED_FLOWS"]=str(len(IDX))
    cmd=[sys.executable,str(inference),
        "--load_model_path",str(ckpt),"--vocab_path",str(vocab),
        "--config_path",str(config),"--test_path",str(subset),
        "--prediction_path",str(prediction),"--labels_num","41",
        "--batch_size","1","--seq_length","128",
        "--embedding","word_pos_seg","--encoder","transformer",
        "--mask","fully_visible","--tokenizer","bert","--output_prob"]
    run_child(cmd,dirs["log"]/f"ETBERT_official_inference_seed{seed}.log",
              child,f"official ETBERT seed{seed}")
    if not prediction.is_file() or not raw.is_file():
        raise RuntimeError("ET-BERT official inference/hook did not produce expected records")
    records=[json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records)!=len(IDX) or sorted(r["sample_ordinal"] for r in records)!=list(range(len(IDX))):
        raise RuntimeError(f"ET-BERT instrumented Classifier calls {len(records)} != {len(IDX)}; no paper data written")
    import platform
    for rec,(idx,row) in zip(records,chosen):
        if rec["sample_count"]!=1 or rec["batch_size"]!=1:
            raise RuntimeError("Native ETBERT inference did not operate at batch_size=1")
        out=dirs["meas"]/f"ET_BERT_cipher_seed{seed}_sample{idx}.json"
        if out.exists():
            existing=json.loads(out.read_text(encoding="utf-8"))
            if existing.get("checkpoint_sha256")!=sha256(ckpt) or existing.get("sample_id")!=int(split[idx]["sample_index"]):
                raise RuntimeError("Existing ET-BERT profile provenance mismatch")
            continue
        result=dict(schema="section56-v17-modelonly-v1",created_utc=datetime.now(timezone.utc).isoformat(),
            model="ET-BERT",dataset="cipher",seed=seed,sample_index=idx,
            sample_id=int(split[idx]["sample_index"]),device="cuda",
            torch_version=rec["torch_version"],cuda_version=torch.version.cuda,
            gpu_name=rec["gpu_name"],platform=platform.platform(),
            source=str(inference),source_sha256=sha256(inference),
            checkpoint=str(ckpt),checkpoint_sha256=sha256(ckpt),
            input_path=str(subset),input_sha256=sha256(subset),
            actual_input_shape=rec["input_shape"],
            input_origin="official_frozen_clean_test_tsv_slice",
            batch_size=1,single_example_per_seed=True,
            warmup=int(env["S56_WARMUP"]),repeats=int(env["S56_REPEATS"]),
            precision="fp32 (native model default; no autocast)",
            includes="Official UER Classifier model forward only; no tokenization",
            excludes=["PCAP parsing","tokenization","CPU-GPU transfer"],
            trainable_parameters=rec["trainable_parameters"],
            total_parameters=rec["total_parameters"],
            parameters_m=rec["total_parameters"]/1e6,
            latency=rec["latency"],flops=rec["flops"],
            paper_eligible=False,blocking_reasons=[
                "Full FLOP operator coverage not attested","GPU model-only latency, not end-to-end throughput"],
            etbert_provenance={
                "official_config_sha256":sha256(config),
                "official_vocab_sha256":sha256(vocab),
                "official_test_full_tsv_sha256":sha256(tsv),
                "official_result_training_sha256":sha256(summary),
                "output_prediction_sha256":sha256(prediction),
                "hook_sha256":sha256(HERE/"etbert_hook"/"sitecustomize.py")})
        pjson(out,result)
    emit("ET-BERT official model-only GPU measurements complete for seed",seed)

def run(args):
    # Sources/checkpoints are resolved when inventory was imported; prevent
    # accidental CLI/environment drift from silently changing their identity.
    if Path(args.code_root).resolve()!=ROOT or Path(args.result_root).resolve()!=RESULT:
        raise RuntimeError("CODE/RESULT path override differs from imported frozen root; set S56_CODE_ROOT and S56_RESULT_ROOT before launching.")
    os.environ["S56_CODE_ROOT"]=str(ROOT)
    os.environ["S56_RESULT_ROOT"]=str(RESULT)
    env=os.environ.copy()
    env["S56_WARMUP"]=str(args.warmup)
    env["S56_REPEATS"]=str(args.repeat)
    if args.warmup<1 or args.repeat<5:raise ValueError("Insufficient profile warmup/repeats")
    dirs={k:HERE/"results"/s for k,s in
          dict(graph="graph_inputs",meas="measurements",log="logs",
               tmp="server_tmp",report="report").items()}
    for d in dirs.values():d.mkdir(parents=True,exist_ok=True)
    state=dict(schema="section56-server-once-v1",start_utc=datetime.now(timezone.utc).isoformat(),
               code_root=str(ROOT),result_root=str(RESULT),model_status={},
               warn=[],failed=[],status="RUNNING",run_hardware={})
    status=HERE/"results"/"run_status.json"
    pjson(status,state)
    emit("START",state["start_utc"],"RESULT",RESULT,"CODE",ROOT)
    torch=check_gpu()
    state["run_hardware"]={"GPU":torch.cuda.get_device_name(0),"torch":str(torch.__version__),
                           "cuda":str(torch.version.cuda)}
    manifest,split=canonical()
    state["canonical_split_sha256"]=sha256(manifest)
    state["selected_sample_ids"]=[int(split[i]["sample_index"]) for i in IDX]
    for model in ORDER:
        emit("MODEL",model)
        try:
            missing=[str(checkpoint_path(model,seed)) for seed in SEEDS
                     if not checkpoint_path(model,seed).is_file()]
            if missing:raise FileNotFoundError("Frozen formal checkpoint absent on GPU: "+", ".join(missing))
            if complete_and_attested(model,dirs,env,split,
                                     state["run_hardware"]["GPU"],
                                     state["run_hardware"]["torch"]):
                state["model_status"][model]="REUSED_VERIFIED_24_OF_24"
                emit("REUSED_VERIFIED_ALL_24",model)
            else:
                if model=="ET-BERT":
                    for seed in SEEDS:etbert_official(seed,dirs,env,split)
                else:
                    if model in ("SMGF-Net","IBGC"):graph_inputs(model,dirs,env)
                    run_pytorch(model,dirs,env)
                state["model_status"][model]="MEASURED"
                emit("MODEL_FINISHED",model)
        except Exception as e:
            msg=f"{model}: {type(e).__name__}: {e}"
            state["model_status"][model]="FAILED_OR_INCOMPLETE"
            state["failed"].append(msg)
            emit("MODEL_FAILED",msg)
            traceback.print_exc()
        finally:
            pjson(status,state)
    try:
        run_child([sys.executable,str(HERE/"build_report.py"),
                   "--measurements",str(dirs["meas"]),"--out",str(dirs["report"]),
                   "--partial-chart"],
                  dirs["log"]/"report_build.log",env,"TableVIII and Fig7")
        state["report_csv"]=str(dirs["report"]/"TableVIII_V17_status.csv")
        state["report_status"]=str(dirs["report"]/"validation_status.json")
    except Exception as e:
        state["failed"].append("Report builder failed: "+str(e))
    state["status"]="COMPLETE_MEASUREMENTS" if not state["failed"] else "PARTIAL_REQUIRES_ATTENTION"
    state["finish_utc"]=datetime.now(timezone.utc).isoformat()
    pjson(status,state)
    emit("FINISHED",state["status"],"status",str(status))
    emit("Table VIII",str(dirs["report"]/"TableVIII_V17_status.csv"))
    emit("Fig.7 (complete)",str(dirs["report"]/"Fig7_V17_latency_robustness.png"))
    emit("Fig.7 (partial, NOT FOR PUBLICATION)",str(dirs["report"]/"Fig7_PARTIAL_NOT_FOR_PUBLICATION.png"))
    emit("IMPORTANT: full FLOPs still require independent operator-coverage verification.")
    if state["failed"]:
        for problem in state["failed"]:emit("ISSUE",problem)
        raise SystemExit(2)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root",default="/home/admin1/lkfiles")
    parser.add_argument("--result-root",default="/home/admin1/lkfiles/RESULT")
    parser.add_argument("--warmup",type=int,default=30)
    parser.add_argument("--repeat",type=int,default=100)
    run(parser.parse_args())

if __name__=="__main__":main()
