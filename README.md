# SMGF_Net
## Environment Setup and Installation

The project is built on Python 3.8+ and the PyTorch ecosystem. Since it involves Graph Neural Networks (GNN) and network packet parsing, please follow these steps to install the dependencies:

**Clone the repository**

git clone [https://github.com/rockylk/SMGF-Net.git](https://github.com/your_username/SMGF-Net.git)
cd SMGF-Net

Install core dependencies
Bash

pip install -r requirements.txt

Special Instructions for PyTorch Geometric (PyG)
If the automatic installation fails, it is recommended to manually install torch-scatter and torch-sparse according to your CUDA version by following the official PyG installation guide.
## Dataset Configuration

This framework supports standard PCAP/PCAPNG format traffic datasets. The recommended directory structure is as follows (supports long-tail/imbalanced classification):
Plaintext

/your_dataset_path/
├── Google/
│   ├── sample1.pcap
│   └── sample2.pcap
├── Netflix/
│   ├── sample3.pcap
│   └── ...
└── YouTube/

Note: The parsing logic for the CipherSpectrum and USTC-TFC2016 datasets utilized in the manuscript's experiments is fully integrated into core/features.py.
Execution and Evaluation

You only need to modify PCAP_DIR (dataset path) and RESULT_DIR (output path for results) in main.py, and then execute the main script:
Bash

python main.py

After execution, the system will automatically run the 6 core experiments mentioned in the manuscript sequentially and export the quantitative data to CSV files:

    Exp 1: Baseline classification performance on clean traffic.

    Exp 2: Robustness baseline under traditional physical obfuscation (Padding, Jitter, Dummy).

    Exp 3: Extreme stress test under semantic-preserved Adaptive Attacks.

    Exp 4: Ablation study of dynamic gating and contrastive optimization.

    Exp 5: Automatic t-SNE dimensionality reduction to generate latent space distribution comparison plots (exp5_tsne_robustness.png).

    Exp 6: Evaluation of line-rate deployment overhead, trainable parameters, and inference latency.

 ## Citation

If you use the code or concepts from this project in your research, please cite our work published in IEEE Transactions on Information Forensics and Security (TIFS) using the following standard format:
代码段

@article{liang2026adversarially,
  title={Adversarially Robust Encrypted Traffic Classification via Semantics-Guided Gated Fusion and Contrastive Optimization},
  author={Liang, Kai and Li, Chuanfeng and Li, Yuanbo and Qian, Lei},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  publisher={IEEE}
}
