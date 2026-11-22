"""Reempaqueta web/public/lab/brain.bin separando los bytes de los arreglos grandes en planos (comprime ~20% mejor)."""
import gzip
import json
import struct
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "web/public/lab/brain.bin"
SHUFFLE = {"cols", "cnt"}


def pack(path=OUT):
    raw = gzip.decompress(path.read_bytes())
    hl = struct.unpack("<I", raw[:4])[0]
    h = json.loads(raw[4:4 + hl])
    base = 4 + hl
    blob = bytearray()
    arrays = {}
    for k, (o, t, n, *rest) in h["arrays"].items():
        a = np.frombuffer(raw, dtype=np.dtype(t), count=n, offset=base + o)
        shuf = bool(rest and rest[0]) or k in SHUFFLE
        data = a.view(np.uint8).reshape(-1, a.itemsize).T.tobytes() if shuf and a.itemsize > 1 and not (rest and rest[0]) else a.tobytes()
        while len(blob) % 8:
            blob.append(0)
        arrays[k] = [len(blob), t, n, 1 if shuf and a.itemsize > 1 else 0]
        blob += data
    h["arrays"] = arrays
    hj = json.dumps(h).encode()
    pad = (8 - (4 + len(hj)) % 8) % 8
    out = struct.pack("<I", len(hj) + pad) + hj + b" " * pad + bytes(blob)
    path.write_bytes(gzip.compress(out, 9))
    print(f"{path.name}: {len(out) / 1e6:.1f} MB → {path.stat().st_size / 1e6:.1f} MB comprimido")


if __name__ == "__main__":
    pack()
