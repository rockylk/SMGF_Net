# -*- coding: utf-8 -*-
"""Instrumentation of the *official* ET-BERT inference's Classifier forward.

Loaded only in child process through PYTHONPATH+S56_ET_BERT_HOOK_OUT. Do NOT
change preprocessing, architecture, weights, or saved test predictions.
"""
import json, os, statistics, time
from pathlib import Path

OUT=os.environ.get("S56_ET_BERT_HOOK_OUT")
if OUT:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("ET-BERT benchmark requires the CUDA GPU")
    if torch.is_grad_enabled():
        torch.set_grad_enabled(False)
    WARMUP=int(os.environ.get("S56_WARMUP","30"))
    REPEATS=int(os.environ.get("S56_REPEATS","100"))
    EXPECTED=int(os.environ.get("S56_EXPECTED_FLOWS","8"))
    LOG=Path(OUT)
    LOG.parent.mkdir(parents=True,exist_ok=True)
    if LOG.exists(): raise RuntimeError("Refusing existing ET-BERT hook output: "+str(LOG))
    ORIGINAL=torch.nn.Module._call_impl
    BUSY=[False]
    COUNT=[0]
    def observed(self,*args,**kwargs):
        # Exact upstream fine-tuning/inference Model class is "Classifier";
        # never measure other internal nn.Modules individually.
        if BUSY[0] or self.__class__.__name__!="Classifier":
            return ORIGINAL(self,*args,**kwargs)
        if COUNT[0]>=EXPECTED:
            raise RuntimeError("Official ET-BERT test loader produced too many Classifier forwards")
        BUSY[0]=True
        try:
            if self.training: raise RuntimeError("ET-BERT official inference Classifier is still in TRAIN mode")
            if not all(p.device.type=="cuda" for p in self.parameters()):
                raise RuntimeError("ET-BERT official Classifier not entirely on CUDA")
            if any(p.dtype not in (torch.float32,torch.int64,torch.int32) for p in self.parameters()):
                raise RuntimeError("ET-BERT must run FP32 for a matched comparison")
            with torch.inference_mode():
                # Native official forward, same original actual test tensors.
                for _ in range(WARMUP): result=ORIGINAL(self,*args,**kwargs)
                torch.cuda.synchronize()
                ms=[]
                for _ in range(REPEATS):
                    first=torch.cuda.Event(enable_timing=True)
                    last=torch.cuda.Event(enable_timing=True)
                    first.record();result=ORIGINAL(self,*args,**kwargs);last.record()
                    torch.cuda.synchronize()
                    ms.append(float(first.elapsed_time(last)))
                # Unknown FLOP operation coverage: keep unaccounted keys.
                with torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                    with_flops=True,record_shapes=True) as prof:
                    ORIGINAL(self,*args,**kwargs)
                    torch.cuda.synchronize()
                events=prof.key_averages()
                counted=[dict(operator=e.key,flops=int(e.flops),count=int(e.count))
                         for e in events if int(e.flops or 0)>0]
                missing=[dict(operator=e.key,count=int(e.count))
                         for e in events if int(e.flops or 0)==0 and e.key.startswith("aten::")]
                flop=sum(x["flops"] for x in counted)
            parameters=sum(p.numel() for p in self.parameters() if p.requires_grad)
            if not ms or min(ms)<=0 or parameters<=0:
                raise RuntimeError("Invalid official ET-BERT profile")
            row=dict(sample_ordinal=COUNT[0],sample_count=1,
                     trainable_parameters=int(parameters),
                     total_parameters=sum(p.numel() for p in self.parameters()),
                     latency=dict(samples_ms=ms,median_ms=statistics.median(ms),
                                  mean_ms=statistics.mean(ms),
                                  std_ms=statistics.stdev(ms) if len(ms)>1 else 0.,
                                  min_ms=min(ms),max_ms=max(ms)),
                     flops=dict(supported_op_flops=flop,
                                supported_op_mflops=flop/1e6,
                                coverage="UNVERIFIED_SUBTOTAL_NOT_TOTAL",
                                counted_ops=counted,unaccounted_ops=missing),
                     input_shape=[list(x.shape) for x in args if torch.is_tensor(x)],
                     gpu_name=torch.cuda.get_device_name(),
                     torch_version=str(torch.__version__),
                     batch_size=1,
                     timing_method="CUDA_events_synced_official_forward",
                     warmup=WARMUP,repeats=REPEATS)
            with LOG.open("a",encoding="utf-8") as f:
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
                f.flush()
            COUNT[0]+=1
            return result
        finally:
            BUSY[0]=False
    torch.nn.Module._call_impl=observed
