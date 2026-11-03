"""Saving and loading results.

Every file is written to a temporary name and then atomically renamed, so a job killed at the 24-hour limit never
leaves a half-written result, and `exists` is a reliable "this task is done" test when the job is resubmitted.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _atomic_write(path: Path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_npz(path, **arrays) -> None:
    def write(tmp):
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
    _atomic_write(Path(path), write)


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_json(path, obj) -> None:
    _atomic_write(Path(path), lambda tmp: Path(tmp).write_text(json.dumps(obj, indent=1, default=_json_default)))


def load_json(path):
    return json.loads(Path(path).read_text())


def exists(path) -> bool:
    return Path(path).exists()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")
