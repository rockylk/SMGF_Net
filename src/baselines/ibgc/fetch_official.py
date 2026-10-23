"""Optional: fetch checksum-pinned upstream source for differential tests.

The baseline itself runs offline without these files. No third-party .pt files
are fetched or loaded. The upstream repository has no license file at the pin;
we distribute source URLs/checksums and our reimplementation, not its full source.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("official_sources/ibgc"))
    args = p.parse_args()
    manifest = json.loads(Path(__file__).with_name("official_source_manifest.json").read_text())
    for rel, expected in manifest["files"].items():
        url = "https://raw.githubusercontent.com/martinal-lee/ibgc/" + manifest["commit"] + "/" + rel
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Upstream hash mismatch: " + rel)
        dest = args.out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(rel, expected)


if __name__ == "__main__":
    main()
