# -*- coding: utf-8 -*-
"""Self-contained negative/positive tests. Synthetic fixtures NEVER used as results."""
import json, tempfile, unittest
from pathlib import Path
from inventory import SEEDS, ROBUST, sha256
from build_report import audit, ORDER

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="s56_TEST_ONLY_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root / "fixture_source.py"
        self.ckpt = self.root / "fixture_ckpt.pt"
        self.inp = self.root / "fixture_input.npz"
        for name in (self.src, self.ckpt, self.inp):
            name.write_bytes(b"FAKE UNIT-TEST DATA - NOT A REAL CHECKPOINT")
        self.sha = {x.name:sha256(x) for x in (self.src, self.ckpt, self.inp)}

    def fixture(self, model, seed, idx, gpu="TEST GPU", device="cuda"):
        # Deliberately fake fixture: audit tests only; never becomes actual
        # measurements and never read by profile_one/run_all.
        return {
            "schema":"section56-v17-modelonly-v1","model":model,
            "dataset":"cipher","seed":seed,"sample_index":idx,"sample_id":idx+100,
            "source":str(self.src),"source_sha256":self.sha[self.src.name],
            "checkpoint":str(self.ckpt),"checkpoint_sha256":self.sha[self.ckpt.name],
            "input_path":str(self.inp),"input_sha256":self.sha[self.inp.name],
            "latency":{"median_ms":0.3+idx/8200,"mean_ms":0.3},
            "flops":{"coverage":"UNVERIFIED_SUBTOTAL_NOT_TOTAL",
                     "supported_op_mflops":23.4},
            "trainable_parameters":123456,"device":device,"gpu_name":gpu,
            "torch_version":"test","precision":"fp32 (native model default; no autocast)",
            "warmup":50,"repeats":200,"batch_size":1,
        }

    def dump(self, rec, label):
        f=self.root/(label+".json")
        f.write_text(json.dumps(rec),encoding="utf-8")
        return f

    def test_empty_refuses_chart(self):
        rows,data,reasons,hardware,chart=audit([],8)
        self.assertFalse(chart)
        self.assertEqual(len(rows),5)
        self.assertEqual(rows[0]["Status"],"MISSING")
        self.assertTrue(any("missing seed" in s for s in reasons))

    def test_120_measured_fixtures_pass_latency_audit_only(self):
        paths=[]
        for model in ORDER:
            for seed in SEEDS:
                for idx in (0,100,500,1000,1500,2000,4000,6000):
                    paths.append(self.dump(self.fixture(model,seed,idx),f"{model}_{seed}_{idx}"))
        rows,by,reasons,hardware,chart=audit(paths,8)
        self.assertTrue(chart)
        self.assertEqual(len(rows),5)
        self.assertTrue(all(x["Status"]=="MEASURED_FLOPS_UNVERIFIED" for x in rows))
        self.assertTrue(all("SUBTOTAL" in x["Supported-Op FLOPs Subtotal (M)"] for x in rows))

    def test_missing_flow_prevents_complete(self):
        files=[]
        for model in ORDER:
            for seed in SEEDS:
                for idx in (0,100,500,1000,1500,2000,4000,6000):
                    if (model,seed,idx)==("IBGC",43,500):continue
                    files.append(self.dump(self.fixture(model,seed,idx),f"{model}_{seed}_{idx}"))
        _,_,reasons,_,chart=audit(files,8)
        self.assertFalse(chart)
        self.assertTrue(any("fewer than" in x for x in reasons))

    def test_changed_checkpoint_rejected(self):
        f=self.dump(self.fixture("SMGF-Net",42,0),"measure")
        self.ckpt.write_bytes(b"MUTATED")
        with self.assertRaisesRegex(ValueError,"changed"):
            audit([f],8)

    def test_wrong_hardware_prevents_chart(self):
        files=[]
        for model in ORDER:
            for seed in SEEDS:
                for idx in (0,100,500,1000,1500,2000,4000,6000):
                    name=f"{model}_{seed}_{idx}"
                    gpu="GPU_X" if model=="SMGF-Net" else "GPU_Y"
                    files.append(self.dump(self.fixture(model,seed,idx,gpu=gpu),name))
        _,_,reasons,_,chart=audit(files,8)
        self.assertFalse(chart)
        self.assertTrue(any("not matched" in s for s in reasons))

    def test_duplicate_rejected(self):
        rec=self.fixture("SMGF-Net",42,0)
        x=self.dump(rec,"a");y=self.dump(rec,"b")
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            audit([x,y],8)

    def test_mismatched_actual_sample_id_rejected(self):
        files=[]
        for model in ORDER:
            for seed in SEEDS:
                for idx in (0,100,500,1000,1500,2000,4000,6000):
                    row=self.fixture(model,seed,idx)
                    if model=="IBGC" and seed==42 and idx==100:
                        row["sample_id"]=999999
                    files.append(self.dump(row,f"{model}_{seed}_{idx}"))
        _,_,reasons,_,chart=audit(files,8)
        self.assertFalse(chart)
        self.assertTrue(any("Sample identity mismatch" in r for r in reasons))

    def test_wrong_floops_scope_rejected(self):
        rec=self.fixture("SMGF-Net",42,0)
        rec["flops"]["coverage"]="COMPLETE"
        with self.assertRaisesRegex(ValueError,"scope"):
            audit([self.dump(rec,"a")],8)

    def test_paper_known_robacc_endpoints(self):
        self.assertEqual(ROBUST["SMGF-Net"],(78.92,86.88))
        self.assertEqual(ROBUST["IBGC"],(17.62,17.37))

if __name__=="__main__":
    unittest.main()