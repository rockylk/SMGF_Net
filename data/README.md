# Data layout

`splits/cipher.json` and `splits/ustc.json` are the exact archived split manifests, copied without modification. They encode class mappings, sample identities, labels, relative PCAP paths, and split membership.

Raw PCAPs, parsed tensor caches, full probability dumps, and trained checkpoints are external assets. They are not bundled or downloadable through an invented project URL. Use the original authorized datasets and preserve the archived relative paths.

Set local paths in `configs/paths.local.json` or supply the corresponding command-line options. See [PROTOCOL.md](../docs/PROTOCOL.md) for sample counts and [REPRODUCING.md](../docs/REPRODUCING.md) for execution instructions.
