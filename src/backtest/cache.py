"""Disk cache of build_inputs per rebalance date (the slow step, about 10 to 60 s a date).

The key covers everything Params depends on: asset types, H, W, the src/params source and the processed data files
(name, size, mtime). Change any of them and a fresh folder is used, so stale estimates are never read.
"""
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import pandas as pd

from data_cleaning import config as C
from src.params import build_inputs

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2] / "results" / "cache" / "params"
PARAMS_SRC = Path(__file__).resolve().parents[1] / "params"


def _fingerprint() -> str:
    h = hashlib.sha256()
    for f in sorted(PARAMS_SRC.glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    for f in sorted(C.PROCESSED_DIR.glob("*.parquet")):
        st = f.stat()
        h.update(f"{f.name}{st.st_size}{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


class ParamsCache:
    """get(d) returns build_inputs(d, W, H, data_for(d)), from memory, then disk, else built and saved."""

    def __init__(self, asset_types, H: int, W: float, root: Path | None = ROOT):
        self.key = dict(asset_types=list(asset_types), H=int(H), W=float(W), source=_fingerprint())
        digest = hashlib.sha256(json.dumps(self.key, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = None if root is None else Path(root) / digest
        self.mem: dict = {}
        self.built = 0

    def get(self, d, data_fn):
        d = pd.Timestamp(d)
        if d in self.mem:
            return self.mem[d]
        f = self.dir / f"{d.date()}.pkl" if self.dir else None
        if f is not None and f.exists():
            P = pickle.loads(f.read_bytes())
        else:
            t = time.time()
            P = build_inputs(d, self.key["W"], H=self.key["H"], data=data_fn(d))
            self.built += 1
            log.info("params %s: %d bonds, %.0fs", d.date(), len(P.cusips), time.time() - t)
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                (f.parent / "key.json").write_text(json.dumps(self.key, indent=2))
                f.write_bytes(pickle.dumps(P))
        self.mem[d] = P
        return P
