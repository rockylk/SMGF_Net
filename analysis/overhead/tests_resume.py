# -*- coding: utf-8 -*-
"""Resume guard tests use temporary dummy evidence, never paper measurements."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import server_oneclick as run
from inventory import SEEDS

class VerifyCompletedMeasurements(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="s56_resume_test_only_")
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.cps={}
        for seed in SEEDS:
            path=self.root/f"checkpoint_{seed}.pt"
            path.write_bytes(f"FAKE SEED {seed}".encode())
            self.cps[seed]=path
        self.source=self.root/"model_source.py"
        self.source.write_bytes(b"FAKE SOURCE")
        self.inp=self.root/"test_sample.pt"
        self.inp.write_bytes(b"FAKE INPUT")
        self.meas=self.root/"measurements"
        self.meas.mkdir()
        self.split=[dict(sample_index=i+101) for i in range(8200)]
        self.env={"S56_WARMUP":"30","S56_REPEATS":"100"}
        for seed in SEEDS:
            for idx in run.IDX:
                ckpt=self.cps[seed]
                payload=dict(schema="section56-v17-modelonly-v1",
                    model="IBGC",dataset="cipher",seed=seed,sample_index=idx,
                    sample_id=self.split[idx]["sample_index"],
                    device="cuda",gpu_name="FAKE_GPU",
                    torch_version="fake_torch",batch_size=1,warmup=30,repeats=100,
                    checkpoint=str(ckpt),checkpoint_sha256=run.sha256(ckpt),
                    source=str(self.source),source_sha256=run.sha256(self.source),
                    input_path=str(self.inp),input_sha256=run.sha256(self.inp),
                    latency=dict(median_ms=0.1),
                    flops=dict(coverage="UNVERIFIED_SUBTOTAL_NOT_TOTAL"))
                filename=self.meas/f"IBGC_cipher_seed{seed}_sample{idx}.json"
                filename.write_text(json.dumps(payload),encoding="utf-8")
        patcher=patch.object(run,"checkpoint_path",side_effect=lambda model,seed:self.cps[seed])
        patcher.start()
        self.addCleanup(patcher.stop)
    def check(self):
        return run.complete_and_attested("IBGC",dict(meas=self.meas),self.env,
                 self.split,"FAKE_GPU","fake_torch")
    def test_verified_24_profiles_skip(self):
        self.assertTrue(self.check())
    def test_missing_one_flow_requires_resume(self):
        (self.meas/"IBGC_cipher_seed42_sample0.json").unlink()
        self.assertFalse(self.check())
    def test_changed_input_evidence_requires_resume(self):
        self.inp.write_bytes(b"MUTATED")
        self.assertFalse(self.check())
    def test_gpu_mismatch_requires_resume(self):
        one=self.meas/"IBGC_cipher_seed44_sample6000.json"
        payload=json.loads(one.read_text(encoding="utf-8"))
        payload["gpu_name"]="DIFFERENT_GPU"
        one.write_text(json.dumps(payload),encoding="utf-8")
        self.assertFalse(self.check())
if __name__=="__main__":unittest.main()
