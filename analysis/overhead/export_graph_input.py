#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export multiple authentic frozen V17 test graph tensors per dataset load."""
from __future__ import annotations
import argparse,importlib.util,json,os,sys,tempfile
from pathlib import Path
from datetime import datetime,timezone
from inventory import ROOT,RESULT,sha256,source_path,base_dir

def module(file,name):
    spec=importlib.util.spec_from_file_location(name,file)
    if not spec or not spec.loader:raise RuntimeError(file)
    obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj

def ibgc(indices,args,torch):
    base=base_dir()
    graph_file=base/"ibgc_graph.py"
    common_file=base/"baseline_v17_common.py"
    if not graph_file.is_file() or not common_file.is_file():
        raise FileNotFoundError("Bundled native IBGC source absent: "+str(base))
    sys.path.insert(0,str(base))
    g=module(graph_file,"s56_ibgc_graph")
    c=module(common_file,"s56_ibgc_common")
    formal=RESULT/"IBGC_V17_V1_100EP"
    prep=json.loads((formal/"cipher"/"preparation.json").read_text(encoding="utf-8"))
    expected=prep["identity"]["raw_cache_sha256"]["test"]
    if not args.raw_test_cache:raise ValueError("IBGC requires --raw-test-cache from original GPU")
    cache=Path(args.raw_test_cache).resolve()
    if not cache.is_file() or sha256(cache)!=expected:
        raise RuntimeError("Frozen test-cache missing or SHA mismatch")
    seqs,y,ids=c.load_raw_partition(cache)
    if len(seqs)!=8200:raise RuntimeError("IBGC wrong test population")
    for i in indices:
        x,e=g.build_graph(seqs[i])
        if x.ndim!=2 or x.shape[1]!=13 or e.shape[0]!=2:
            raise RuntimeError("IBGC graph representation drift")
        data=dict(x=torch.from_numpy(x).float(),
                  edge_index=torch.from_numpy(e).long(),
                  batch=torch.zeros(x.shape[0],dtype=torch.long))
        yield data,dict(model="IBGC",index=i,sample_id=int(ids[i]),
                        source="Frozen raw-test cache plus ORIGINAL ibgc_graph.build_graph",
                        raw_cache_sha256=expected,
                        source_code_sha256=sha256(base/"ibgc_graph.py"))

def smgf(indices,args,torch):
    file=source_path("SMGF-Net")
    if not args.pcap_root or not Path(args.pcap_root).is_dir():
        raise FileNotFoundError("SMGF requires --pcap-root containing original raw CipherSpectrum PCAPs")
    with tempfile.TemporaryDirectory(prefix="s56_smgf_original_test_") as td:
        old={k:os.environ.get(k) for k in ("SMGF_RESULT_DIR","SMGF_PCAP_DIR")}
        os.environ["SMGF_RESULT_DIR"]=td
        os.environ["SMGF_PCAP_DIR"]=str(Path(args.pcap_root).resolve())
        try:
            s=module(file,"s56_original_smgf")
            s.LOGGER=s.Logger(Path(td)/"log.txt")
            s.RUN_DIR=Path(td)
            s.SPLIT_FILE=Path(td)/"split_manifest.json"
            s.CORE.RUN_DIR=Path(td)
            s.CORE.SPLIT_MANIFEST_FILE=s.SPLIT_FILE
            _,_,test,_,_,nc,_,manifest=s.load_and_prepare_data()
            if nc!=41 or len(test)!=8200:raise RuntimeError("SMGF test split drift")
            split=json.loads(s.SPLIT_FILE.read_text(encoding="utf-8"))
            if len(split["test"])!=8200:raise RuntimeError("SMGF test manifest drift")
            from torch_geometric.loader import DataLoader
            for i in indices:
                item=next(iter(DataLoader([test[i]],batch_size=1,shuffle=False)))
                keys=("x","edge_index","edge_attr","batch",
                      "stats_attr","finger_attr","entropy_attr")
                data={k:getattr(item,k).detach().cpu().clone() for k in keys}
                yield data,dict(model="SMGF-Net",index=i,
                                sample_id=int(split["test"][i]["sample_index"]),
                                source="Original V17 loader and train-only scalers",
                                dataset_manifest_sha256=manifest,source_code_sha256=sha256(file))
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,choices=("SMGF-Net","IBGC"))
    ap.add_argument("--indices",default="0,100,500,1000,1500,2000,4000,6000")
    ap.add_argument("--pcap-root")
    ap.add_argument("--raw-test-cache")
    ap.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"results"/"graph_inputs")
    args=ap.parse_args()
    indices=[int(x) for x in args.indices.split(",") if x.strip()]
    if not indices or len(set(indices))!=len(indices) or any(not 0<=i<8200 for i in indices):
        raise ValueError("Invalid or repeated held-out test indices")
    import torch
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data=ibgc(indices,args,torch) if args.model=="IBGC" else smgf(indices,args,torch)
    for tensors,info in data:
        idx=info["index"]
        path=args.output_dir/f"{args.model.replace('-','_')}_cipher_sample{idx}.pt"
        if path.exists():
            meta_path=path.with_suffix(".provenance.json")
            if not meta_path.is_file():raise FileExistsError("Incomplete graph provenance: "+str(path))
            old=json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("model")!=args.model or old.get("index")!=idx or old.get("output_sha256")!=sha256(path):
                raise RuntimeError("Existing graph data do not match provenance: "+str(path))
            print(f"SKIP_VERIFIED {path}",flush=True)
            continue
        torch.save(tensors,path)
        info.update({"output_sha256":sha256(path),"dataset":"CipherSpectrum",
                     "utc":datetime.now(timezone.utc).isoformat(),
                     "shapes":{k:list(v.shape) for k,v in tensors.items()}})
        path.with_suffix(".provenance.json").write_text(
            json.dumps(info,indent=2,ensure_ascii=False),encoding="utf-8")
        print(f"EXPORTED {path} SHA256={info['output_sha256']}",flush=True)
if __name__=="__main__":main()