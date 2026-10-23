# -*- coding: utf-8 -*-
"""Simulated sibling-of-RESULT server topology; test artifacts are NOT paper data."""
from pathlib import Path
import tempfile, unittest, shutil
import inventory

class DirectoryLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="s56_sibling_layout_TEST_")
        self.addCleanup(self.temp.cleanup)
        self.original=inventory.ROOT
        inventory.ROOT=Path(self.temp.name)
        self.addCleanup(lambda:setattr(inventory,"ROOT",self.original))
        (inventory.ROOT/"RESULT").mkdir()
        self.v=inventory.HERE/"vendor"
    def make(self,dirname,files):
        dest=inventory.ROOT/dirname
        dest.mkdir(parents=True,exist_ok=True)
        for f in files:
            output=dest/f
            output.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(self.v/f,output)
        return dest
    def test_sibling_ibgc_exact_source_selected(self):
        folder=self.make("IBGC_V17_V1_100EP",
                         ["ibgc_model.py","ibgc_graph.py","baseline_v17_common.py"])
        self.assertEqual(inventory.base_dir(),folder)
        self.assertEqual(inventory.source_path("IBGC"),folder/"ibgc_model.py")
    def test_ibgc_historical_drift_uses_frozen_source(self):
        folder=self.make("IBGC_V17_V1_100EP",
                         ["ibgc_model.py","ibgc_graph.py","baseline_v17_common.py"])
        (folder/"ibgc_graph.py").write_text("# DRIFT; NOT FORMAL V17",encoding="utf-8")
        self.assertEqual(inventory.base_dir(),self.v)
    def test_pean_model_and_trf_must_both_match(self):
        folder=self.make("PEAN_NHP_V1_100EP",
                         ["PEAN_NHP_V17_model.py","official_sources/PEAN/TRF.py"])
        self.assertEqual(inventory.source_path("PEAN-NHP"),folder/"PEAN_NHP_V17_model.py")
        (folder/"official_sources/PEAN/TRF.py").unlink()
        self.assertEqual(inventory.source_path("PEAN-NHP"),
                         self.v/"PEAN_NHP_V17_model.py")
    def test_smgf_v17_requires_matching_core(self):
        folder=self.make("SMGF-Net",["V17-SMGF-Net_final_paper_experiments.py",
                                      "V16-SMGF-Net_final_paper_experiments.py"])
        self.assertEqual(inventory.source_path("SMGF-Net"),
                         folder/"V17-SMGF-Net_final_paper_experiments.py")
        (folder/"V16-SMGF-Net_final_paper_experiments.py").write_text(
            "# wrong core",encoding="utf-8")
        self.assertEqual(inventory.source_path("SMGF-Net"),
                         self.v/"V17-SMGF-Net_final_paper_experiments.py")
    def test_checkpoint_never_uses_sibling_training_folders(self):
        self.make("PEAN_NHP_V1_100EP",["PEAN_NHP_V17_model.py"])
        ckpt=inventory.checkpoint_path("PEAN-NHP",42)
        self.assertIn("/RESULT/",str(ckpt).replace("\\","/"))
        self.assertNotIn("/PEAN_NHP_V1_100EP/",str(ckpt).replace("\\","/"))
if __name__=="__main__":
    unittest.main()
