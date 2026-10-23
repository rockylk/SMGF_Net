# -*- coding: utf-8 -*-
"""PEAN-NHP V1 data preparation for fair structural-input comparison.

PEAN-NHP = PEAN No Header/Payload.  It intentionally removes ALL model-visible
TCP/UDP header bytes and ALL raw payload bytes while keeping the native PEAN
tensor shapes and the packet-length branch unchanged.

The byte tensor is a traffic-independent constant for every sample, packet
slot, condition, and dataset: [CLS, SEP, PAD, ..., PAD].  Therefore it cannot
encode ports, flags, seq/ack values, protocol/header fields, payload content,
payload length via SEP position, packet count, or attack condition.

The only sample-dependent model input is the first-ten serialized full-packet
length sequence.  Mixed/Adaptive still use the frozen V17 physical attack geometry before the
length view is regenerated. The attack simulator operates on packet size/time
metadata and never reads Raw payload content. No TCP/UDP field value, header byte,
payload byte, or derived byte token is exposed to the PEAN-NHP model.
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Sequence

import numpy as np

PACKETS = 10
BYTES_PER_PACKET = 400
LENGTH_VOCAB = 2000
BYTE_VOCAB = 261
V17_OBSERVATION_CAP = 400
PAD, CLS, SEP, BYTE_OFFSET = 0, 1, 2, 5
ATTACK_BASE_SEED = 20260919
PRIMARY_STRENGTH = 1.0
PROTOCOL = "PEAN-NHP-v1-constantByteView-first10-fullPacketLength-v17-attackSeed20260919"

def check(cond, message):
    if not cond:
        raise RuntimeError(message)

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def stable_seed(*parts, modulo=2**31-1):
    digest=hashlib.sha256("|".join(str(x) for x in parts).encode()).hexdigest()
    return int(digest[:16],16)%modulo

def dump_json(path: Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                   encoding="utf-8")
    os.replace(tmp,path)

def verify_vocabulary(vocab_file: Path):
    rows=vocab_file.read_text(encoding="utf-8-sig").splitlines()
    expected=["[PAD]","[CLS]","[SEP]","[MASK]","[UNK]"]+[f"{i:02x}" for i in range(256)]
    check(rows==expected,"Official PEAN vocabulary drift")
    return sha256_file(vocab_file)

def constant_byte_view() -> np.ndarray:
    """Return the ONLY allowed PEAN-NHP byte tensor.

    Every one of the ten slots is identical and independent of the flow:
    [CLS, SEP, PAD x 398].  SEP is fixed at position 1, so neither payload
    length nor packet serialized length can leak through the byte branch.
    """
    out=np.zeros((PACKETS,BYTES_PER_PACKET),dtype=np.uint16)
    out[:,0]=CLS
    out[:,1]=SEP
    return out

_CONSTANT_BYTE_VIEW=constant_byte_view()

def packet_metadata(packets):
    """Extract only metadata needed by PEAN-NHP and its physical attack.

    No TCP/UDP field value and no Raw payload byte is read. Header length is
    used only to reproduce the minimum serialized size of injected dummy
    packets; it is never exposed to the model.
    """
    from scapy.all import IP,TCP,UDP,Raw
    valid=[p for p in packets if IP in p and (TCP in p or UDP in p)]
    check(len(valid)>=2,"PEAN-NHP requires >=2 valid IP TCP/UDP packets")
    out=[]
    for i,p in enumerate(valid[:V17_OBSERVATION_CAP]):
        size=int(len(p))
        check(size>0,"Nonpositive serialized packet length")
        try:
            t=float(p.time)
        except Exception:
            t=float(i)
        q=p.copy()
        if Raw in q:
            del q[Raw]
        header_len=int(len(bytes(q)))
        check(0 < header_len <= size,
              "Invalid header-length metadata while preparing PEAN-NHP")
        out.append(dict(size=size,time=t,header_len=header_len))
    return out

def encode_metadata(metadata):
    check(len(metadata)>=2,"PEAN-NHP metadata requires >=2 packets")
    first=metadata[:PACKETS]
    x=_CONSTANT_BYTE_VIEW.copy()
    lengths=np.zeros((PACKETS,),dtype=np.uint16)
    clipped=0
    for i,m in enumerate(first):
        size=int(m["size"])
        check(size>0,"Nonpositive serialized packet length")
        clipped += int(size>=LENGTH_VOCAB)
        lengths[i]=np.uint16(min(size,LENGTH_VOCAB-1))
    return x,lengths,dict(real_packets=len(first),lengths_clipped=clipped)

def encode_packet_sequence(packets):
    return encode_metadata(packet_metadata(packets))

def read_clean_pcap(path: Path):
    from scapy.all import rdpcap
    return encode_packet_sequence(rdpcap(str(path)))

# -------------------------------------------------------------------------
# Physical perturbation family: metadata-only reproduction of the same
# padding / IAT-jitter / dummy-insertion geometry used by the PEAN V2 runner.
# Raw payload CONTENT is never read.
# -------------------------------------------------------------------------
def _consume_random_filler(length,np_rng,semantics_preserving):
    """Preserve the native attack RNG stream without materializing payload."""
    if length>0 and not semantics_preserving:
        np_rng.integers(0,256,size=int(length),dtype=np.uint8)

def attack_packet_length_padding_meta(metadata,strength,np_rng,semantics_preserving):
    out=[dict(m) for m in metadata]
    if not out:return out
    k=max(1,int(len(out)*(0.10+0.20*strength)))
    idx=np_rng.choice(len(out),size=min(k,len(out)),replace=False)
    pads=np_rng.integers(20,max(int(64+512*strength),21),size=len(idx))
    for pos,pad_len in zip(idx.tolist(),pads.tolist()):
        out[pos]["size"] += int(pad_len)
        _consume_random_filler(int(pad_len),np_rng,semantics_preserving)
    return out

def _metadata_times(metadata):
    return np.asarray([float(m["time"]) for m in metadata],dtype=np.float64)

def attack_iat_jitter_meta(metadata,strength,np_rng):
    out=[dict(m) for m in metadata]
    n=len(out)
    if n<=1:return out
    times=_metadata_times(out)
    iats=np.diff(times,prepend=times[0]);iats[0]=0.;iats=np.clip(iats,0.,None)
    k=max(1,int(n*(0.20+0.30*strength)))
    idx=np_rng.choice(n,size=min(k,n),replace=False)
    mult=np_rng.uniform(1.-(0.10+0.70*strength),1.+(0.10+0.70*strength),size=len(idx))
    for pos,m in zip(idx.tolist(),mult.tolist()):
        if pos: iats[pos]=max(iats[pos]*float(m),0.)
    rebuilt=np.cumsum(iats);base=times[0]
    for i,m in enumerate(out):m["time"]=base+float(rebuilt[i])
    return out

def attack_dummy_injection_meta(metadata,strength,np_rng,py_rng,semantics_preserving):
    out=[dict(m) for m in metadata]
    n=len(out)
    if not n:return out
    insert_num=max(1,int(n*(0.03+0.10*strength)))
    times=_metadata_times(out)
    if len(times)>1:
        gaps=np.diff(times);positive=gaps[gaps>0]
        med=float(np.median(positive)) if len(positive) else 1e-6
    else: med=1e-6
    positions=sorted(np_rng.integers(1,max(2,n),size=insert_num).tolist())
    offset=0
    for p in positions:
        base_idx=min(max(p-1+offset,0),len(out)-1)
        base=out[base_idx]
        target=int(py_rng.uniform(40,200+400*strength))
        header_len=int(base["header_len"])
        payload_len=max(0,target-header_len)
        _consume_random_filler(payload_len,np_rng,semantics_preserving)
        dummy=dict(
            size=max(header_len,target),
            header_len=header_len,
            time=float(base["time"])+max(med*py_rng.uniform(.2,1.),1e-6),
        )
        out.insert(min(p+offset,len(out)),dummy);offset+=1
    out.sort(key=lambda m:float(m["time"]))
    return out

def apply_attack_metadata(metadata,attack_name,strength,attack_seed):
    np_rng=np.random.default_rng(attack_seed);py_rng=random.Random(attack_seed)
    sem=attack_name.lower()=="adaptive"
    check(attack_name.lower() in ("mixed","adaptive"),f"Unsupported attack {attack_name}")
    out=attack_packet_length_padding_meta(metadata,strength,np_rng,sem)
    out=attack_iat_jitter_meta(out,strength,np_rng)
    out=attack_dummy_injection_meta(out,strength,np_rng,py_rng,sem)
    return out[:V17_OBSERVATION_CAP]

# -------------------------------------------------------------------------
# Dense immutable caches.
# -------------------------------------------------------------------------
def dataset_dir(run_dir:Path,spec):return run_dir/spec.key
def prep_meta_path(run_dir,spec):return dataset_dir(run_dir,spec)/"preparation.json"
def clean_cache_path(run_dir,spec,part):
    return dataset_dir(run_dir,spec)/"prepared"/f"{part}_clean.npz"
def attacked_cache_path(run_dir,spec,attack,strength):
    return dataset_dir(run_dir,spec)/"prepared"/f"test_{attack}_{strength:.1f}.npz"
def progress_path(run_dir,spec):return dataset_dir(run_dir,spec)/"preparation_progress.json"

def save_arrays_atomic(path,x,l,y,indices):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,bytes=x,lengths=l,labels=y,sample_indices=indices)
    os.replace(tmp,path)
    return sha256_file(path)

def validate_arrays(path,sha,rows,num_classes):
    if not path.is_file() or sha256_file(path)!=sha:return False
    try:
        with np.load(path,allow_pickle=False) as z:
            x=z["bytes"];l=z["lengths"];y=z["labels"].astype(np.int64)
            idx=z["sample_indices"].astype(np.int64)
        n=len(rows)
        if x.shape!=(n,PACKETS,BYTES_PER_PACKET) or l.shape!=(n,PACKETS):return False
        if y.shape!=(n,) or idx.shape!=(n,):return False
        if int(x.max())>=BYTE_VOCAB or int(l.max())>=LENGTH_VOCAB:return False
        expected_y=np.asarray([int(r["label"]) for r in rows],dtype=np.int64)
        expected_i=np.asarray([int(r["sample_index"]) for r in rows],dtype=np.int64)
        if not np.array_equal(y,expected_y) or not np.array_equal(idx,expected_i):return False
        if not np.all((y>=0)&(y<num_classes)):return False
        expected=np.broadcast_to(_CONSTANT_BYTE_VIEW,(n,PACKETS,BYTES_PER_PACKET))
        if not np.array_equal(x,expected):return False
        return True
    except Exception:return False

def _identity(spec,vocab_sha,source_sha):
    return dict(dataset=spec.name,key=spec.key,protocol=PROTOCOL,
                split_sha256=spec.expected_split_sha,
                dataset_manifest_sha256=spec.expected_dataset_manifest,
                counts=spec.expected_counts,num_classes=spec.num_classes,
                vocab_sha256=vocab_sha,source_sha256=source_sha,
                attack_base_seed=ATTACK_BASE_SEED,
                attack_translation="V17 physical packet family; PEAN-NHP byte view constant; length view regenerated after attack",
                model_visible_inputs="packet-length sequence only; no header/payload bytes",
                byte_view="constant [CLS,SEP,PADx398] for all 10 slots")

def import_clean_caches_if_requested(spec,run_dir,source_dir,identity,split,logger):
    """PEAN-NHP must never import native-PEAN caches.

    Native PEAN caches contain model-visible TCP/UDP header and payload byte
    tokens, so cache reuse across the two named variants would violate the
    fairness boundary even if labels/splits match.
    """
    if source_dir is not None:
        raise RuntimeError(
            "PEAN-NHP refuses native PEAN cache import; prepare a fresh NHP cache")
    return {}

def prepare_dataset(spec,run_dir,vocab_file,source_sha,logger,clean_import=None):
    import baseline_v17_common as C
    C.validate_dataset_spec(spec)
    split=json.loads(spec.split_manifest.read_text(encoding="utf-8"))
    vocab_sha=verify_vocabulary(vocab_file)
    ident=_identity(spec,vocab_sha,source_sha)
    meta_file=prep_meta_path(run_dir,spec)
    required_names=["train_clean","validation_clean","test_clean","test_mixed_1.0","test_adaptive_1.0"]
    if meta_file.is_file():
        old=json.loads(meta_file.read_text(encoding="utf-8"))
        if all(old.get(k)==v for k,v in ident.items()):
            mapping={
                "train_clean":(clean_cache_path(run_dir,spec,"train"),split["train"]),
                "validation_clean":(clean_cache_path(run_dir,spec,"validation"),split["validation"]),
                "test_clean":(clean_cache_path(run_dir,spec,"test"),split["test"]),
                "test_mixed_1.0":(attacked_cache_path(run_dir,spec,"mixed",1.0),split["test"]),
                "test_adaptive_1.0":(attacked_cache_path(run_dir,spec,"adaptive",1.0),split["test"])}
            if all(name in old.get("cache_sha256",{})
                       and validate_arrays(mapping[name][0],old["cache_sha256"][name],
                                           mapping[name][1],spec.num_classes)
                       for name in required_names):
                logger(f"[PEAN-NHP][{spec.name}] reuse verified full prepared dataset")
                return old
        raise RuntimeError(f"{spec.name}: existing formal preparation has different provenance/SHA")

    base=dataset_dir(run_dir,spec);(base/"prepared").mkdir(parents=True,exist_ok=True)
    prog_file=progress_path(run_dir,spec)
    cache_sha={};reports={}
    if prog_file.is_file():
        prog=json.loads(prog_file.read_text(encoding="utf-8"))
        check(all(prog.get(k)==v for k,v in ident.items()),
              f"{spec.name}: partial preparation provenance differs")
        cache_sha.update(prog.get("cache_sha256",{}));reports.update(prog.get("reports",{}))
    if not cache_sha and clean_import is not None:
        imported=import_clean_caches_if_requested(spec,run_dir,clean_import,ident,split,logger)
        for part,sha in imported.items():cache_sha[f"{part}_clean"]=sha

    jobs=[
        ("train_clean","train",None),
        ("validation_clean","validation",None),
        ("test_clean","test",None),
        ("test_mixed_1.0","test","mixed"),
        ("test_adaptive_1.0","test","adaptive")]
    from scapy.all import rdpcap
    for name,part,attack in jobs:
        rows=split[part]
        path=(clean_cache_path(run_dir,spec,part) if attack is None
              else attacked_cache_path(run_dir,spec,attack,1.0))
        prior=cache_sha.get(name)
        if prior and validate_arrays(path,prior,rows,spec.num_classes):
            logger(f"[PEAN-NHP][prepare][{spec.name}][{name}] reuse {len(rows)}/{len(rows)}")
            continue
        n=len(rows)
        x=np.empty((n,PACKETS,BYTES_PER_PACKET),dtype=np.uint16)
        l=np.empty((n,PACKETS),dtype=np.uint16)
        y=np.empty(n,dtype=np.int64);idx=np.empty(n,dtype=np.int64)
        clipped=0;padded=0;byte_changed=0;length_changed=0
        for i,row in enumerate(rows):
            p=spec.root/row["relative_path"]
            check(p.is_file(),f"Missing frozen PCAP {p}")
            packets=rdpcap(str(p))
            clean_meta=packet_metadata(packets)
            if attack is None:
                xb,xl,info=encode_metadata(clean_meta)
            else:
                aseed=stable_seed(ATTACK_BASE_SEED,spec.name,attack,"1.000",
                                  int(row["sample_index"]))
                attacked_meta=apply_attack_metadata(clean_meta,attack,1.0,aseed)
                xb,xl,info=encode_metadata(attacked_meta)
                # Fairness audit: byte view MUST remain identical under every attack.
                cxb,cxl,_=encode_metadata(clean_meta)
                byte_changed += int(not np.array_equal(xb,cxb))
                length_changed += int(not np.array_equal(xl,cxl))
            x[i],l[i]=xb,xl;y[i]=int(row["label"]);idx[i]=int(row["sample_index"])
            clipped+=info["lengths_clipped"];padded+=info["real_packets"]<PACKETS
            if (i+1)%500==0 or i+1==n:
                logger(f"[PEAN-NHP][prepare][{spec.name}][{name}] {i+1}/{n}")
        cache_sha[name]=save_arrays_atomic(path,x,l,y,idx)
        reports[name]=dict(n=n,padded_flow_lt10=padded,
                           clipped_packet_lengths_ge2000=clipped,
                           byte_view_changed_n=(byte_changed if attack else None),
                           length_view_changed_n=(length_changed if attack else None))
        dump_json(prog_file,dict(ident,cache_sha256=cache_sha,reports=reports,
                                  updated_at=time.strftime("%Y-%m-%dT%H:%M:%S")))
        if attack is not None:
            check(byte_changed==0,
                  f"PEAN-NHP fairness violation: attack changed constant byte view in {name}")
        logger(f"[PEAN-NHP][prepare][{spec.name}][{name}] saved sha={cache_sha[name]}")
    meta=dict(ident,cache_sha256=cache_sha,reports=reports,
              created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    dump_json(meta_file,meta)
    if prog_file.is_file():prog_file.unlink()
    return meta

def load_condition(run_dir,spec,metadata,condition):
    if condition in ("train","validation","test"):
        name=f"{condition}_clean";path=clean_cache_path(run_dir,spec,condition)
    elif condition in ("mixed_1.0","adaptive_1.0"):
        name=f"test_{condition}";attack=condition.split("_")[0]
        path=attacked_cache_path(run_dir,spec,attack,1.0)
    else:raise ValueError(condition)
    rows_key=("test" if condition in ("mixed_1.0","adaptive_1.0") else condition)
    expected_n=spec.expected_counts[rows_key]
    check(path.is_file() and sha256_file(path)==metadata["cache_sha256"][name],
          f"{spec.name} {condition}: missing/SHA-mismatched cache")
    with np.load(path,allow_pickle=False) as z:
        x=z["bytes"].copy();l=z["lengths"].copy();y=z["labels"].astype(np.int64)
        idx=z["sample_indices"].astype(np.int64)
    check(x.shape==(expected_n,PACKETS,BYTES_PER_PACKET),"Byte cache shape drift")
    check(l.shape==(expected_n,PACKETS) and y.shape==(expected_n,)
          and idx.shape==(expected_n,),"Length/label/index cache shape drift")
    check(np.isfinite(l).all() and np.all((y>=0)&(y<spec.num_classes)),
          "Prepared PEAN cache invalid")
    return x,l,y,idx
