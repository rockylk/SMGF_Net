"""Atomic artifacts and strict, immutable provenance checks."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    os.replace(temp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def load_checkpoint(path, identity):
    # Only own trusted checkpoints; never load downloaded third-party pickles.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("schema_version") != 1 or obj.get("identity") != identity:
        raise RuntimeError("Checkpoint schema/provenance mismatch: " + str(path))
    return obj


def bind_manifest(path, identity):
    path = Path(path)
    if path.exists():
        if read_json(path) != identity:
            raise RuntimeError("Existing run has different provenance: " + str(path))
    else:
        write_json(path, identity)


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, text):
        import time
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(text)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
