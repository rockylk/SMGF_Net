#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU layout preflight. No torch; no model deserialization; no training."""
from inventory import ROOT, RESULT, HERE, source_path, base_dir, checkpoint_path, SEEDS
MODELS=("SMGF-Net","USTC-CNN","ET-BERT","PEAN-NHP","IBGC")
def main():
    print("=== Section 5.6 SERVER LAYOUT ===")
    print("CODE_ROOT:", ROOT)
    print("RESULT_ROOT:", RESULT)
    print("SIBLING DIRECTORIES:")
    for name in ("ET-BERT","IBGC_V17_V1_100EP","PEAN_NHP_V1_100EP",
                 "PEAN_FINAL_V17","PEAN_FINAL_V2_100EP","RESULT"):
        print(" ",name,"PRESENT" if (ROOT/name).is_dir() else "ABSENT")
    missing=0
    for model in MODELS:
        file=source_path(model)
        print(f"{model}: CODE={'OK' if file.is_file() else 'MISSING'}"
              f" SOURCE={'FROZEN_VENDOR' if file.is_relative_to(HERE/'vendor') else 'VERIFIED_SERVER'}"
              f" PATH={file}")
        missing+=not file.is_file()
        for seed in SEEDS:
            ckpt=checkpoint_path(model,seed)
            print(f"  seed{seed}: {'PRESENT' if ckpt.is_file() else 'MISSING'} {ckpt}")
            missing+=not ckpt.is_file()
    print("IBGC SOURCE DIR:",base_dir())
    print("ABSENT_FILES:",int(missing))
    print("No GPU measurement run by this script.")
if __name__=="__main__":main()
