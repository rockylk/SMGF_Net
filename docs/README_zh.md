# GitHub 源码归档说明

本目录整理正式 SMGF-Net V17 与 USTC-CNN、ET-BERT、PEAN-NHP、IBGC 四个基线的最终实验代码、固定数据划分、结果快照和分析代码。原实验目录保持不变。

## 先看哪些文件

- [完整实验结果](../results/RESULTS.md)：Clean、Mixed、Adaptive、Macro OvR AUROC、消融、门控分析与计算开销。
- [强度敏感性结果](../results/SENSITIVITY.md)：四模型在 CipherSpectrum 上各强度的真实测量结果。
- [复现说明](REPRODUCING.md)：原始资产要求、运行命令、输出位置与已有归档的边界。
- [实验协议](PROTOCOL.md)：数据集、划分、训练设置、基线输入适配与统计单位。
- [第三方来源](THIRD_PARTY.md)：官方仓库、固定 commit、已有源码及许可证状态。

## 不需要 GPU 的核验

```bash
python scripts/validate_archive.py
python scripts/summarize_results.py --check
```

重新生成结果 Markdown 和 CSV：

```bash
python scripts/summarize_results.py
```

结果由最终原始逐 seed 文件自动汇总，而非从论文中的四舍五入数值手工重抄。主比较为三个独立训练模型的均值和样本 SD；消融固定 seed42，不混作三模型统计。

## 为什么仍保留原始文件名

最终运行器会对 V16 核心、V17 模型和部分基线源码做 SHA-256 校验。随意改名、改内容或转换换行符会破坏依赖与实验追溯。本归档按模型、分析任务、配置、数据和结果组织目录，将受哈希约束的源文件保留在对应子目录，另提供统一路径适配入口。

最终配置是 learned Gate + attacked-label CE，KL 权重为 0，鲁棒 batch 比例为 0.20。不能直接运行历史 V17 基础文件的默认主程序来替代最终实验。

## 运行前需要补齐的原始资产

1. 对应固定划分的原始 PCAP 数据。
2. ET-BERT 的固定官方源码版本、预训练权重和词表。
3. 复现 CipherSpectrum 原始三 seed 编排及匹配消融时所需的正式 seed42 checkpoint。

这些大文件未打入源码目录，也没有把缺少的权重伪装成可从零复现的完整资产包。USTC 三 seed 训练可以通过统一入口顺序执行。设置路径时复制 `configs/paths.example.json` 为 `configs/paths.local.json`，填写真实本地路径。

归档没有包含手稿、SSH 连接信息、密码、缓存、旧调参目录和带“云端”字样的副本。保留的原始 manifest 中可能含历史实验路径，用于追溯，并不用于自动连接服务器。

本次工作只建立待上传的本地目录，没有执行 GitHub 推送。
