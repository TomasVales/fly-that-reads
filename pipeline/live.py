"""
Laboratorio en vivo: entrena la mosca sobre el cerebro que corre en el navegador, mide lesiones y exporta todo.

Cerebro: las neuronas que pueden activarse desde el olfato Y que llegan a las células de Kenyon. El resto nunca se
activa o nunca influye en lo que escribe, así que sacarlas no cambia nada (es exacto, no una aproximación).

Lesiones: la mosca aprende con el cerebro sano; después se "apaga" un grupo de neuronas (su salida queda en 0) y se
mide cómo escribe. Igual que un experimento de lesión en un laboratorio real.

Uso: python pipeline/live.py [--train 30000] → web/public/lab/brain.bin + web/public/data/lab.json
"""
import argparse
import gzip
import json
import math
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from books import PUNCT, load_all  # noqa: E402
from connectome import DATA, INHIBITORY, load_connectome  # noqa: E402
from run import DEV, FAMOUS, Graph, Readout, log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
np.seterr(all="ignore")


def hops(M, seeds, N, k=60):
    d = np.full(N, 99, np.int16)
    d[seeds] = 0
    f = np.zeros(N, np.float32)
    f[seeds] = 1
    for h in range(1, k + 1):
        nf = (M @ f > 0) & (d == 99)
        d[nf] = h
        f = nf.astype(np.float32)
        if not nf.any():
            break
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=30_000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=3000)
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--ticks", type=int, default=3)
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    books = load_all()
    vocab = sorted(set("".join(b["text"] for b in books)))
    idx = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    enc = lambda s: np.array([idx[c] for c in s if c in idx], np.int64)  # noqa: E731
    nb = len(books)

    # ---------- cerebro ----------
    C = load_connectome(rng=np.random.default_rng(a.seed))
    Wf = C["W"].tocsr()
    Nf = Wf.shape[0]
    A = (Wf != 0).astype(np.float32).tocsr()
    down = hops(A, C["inputs"], Nf)
    up = hops(A.T.tocsr(), C["outputs_pref"], Nf)
    keep = np.where((down < 99) & (up < 99))[0]
    remap = np.full(Nf, -1)
    remap[keep] = np.arange(len(keep))
    W = Wf[keep][:, keep].tocsr()
    W.sort_indices()
    N = W.shape[0]
    # conteo de sinapsis con signo (lo que viaja al navegador) → mismo peso exacto acá y allá
    cnt = np.rint(np.sign(W.data) * np.expm1(np.abs(W.data))).astype(np.int64)
    cnt = np.clip(cnt, -32767, 32767).astype(np.int16)
    W.data = (np.sign(cnt) * np.log1p(np.abs(cnt.astype(np.float32))) * a.scale).astype(np.float32)
    inputs = remap[C["inputs"]]
    inputs = inputs[inputs >= 0]
    KC = remap[C["outputs_pref"]]
    KC = KC[KC >= 0]
    log(f"cerebro del laboratorio: {N:,} neuronas · {W.nnz:,} conexiones · {len(inputs):,} olfativas · {len(KC):,} Kenyon")

    # ---------- grupos para lesionar ----------
    cl = pd.read_csv(next(DATA.glob("*classif*")), low_memory=False)
    rid = C["ids"][keep]
    info = cl.set_index(cl["root_id"].astype("int64")).reindex(rid)
    sup = info["super_class"].astype(str).to_numpy()
    klass = info["class"].astype(str).to_numpy()
    side = info["side"].astype(str).to_numpy()
    colsign = np.zeros(N)
    Wc = W.tocsc()
    for j in range(N):
        d = Wc.data[Wc.indptr[j]:Wc.indptr[j + 1]]
        colsign[j] = np.sign(d).sum() if len(d) else 0
    inhib = colsign < 0
    # APL: la neurona inhibidora gigante del cuerpo fungiforme (una por lado); se reconoce porque le habla a casi todas las Kenyon
    isKC = np.zeros(N, bool)
    isKC[KC] = True
    kc_targets = np.asarray((Wc[isKC] != 0).sum(0)).ravel()
    cand = np.where(inhib)[0]
    top = cand[np.argsort(-kc_targets[cand])][:4]
    apl = top[kc_targets[top] > 800]
    log("APL (candidatas): " + ", ".join(f"{i}: {kc_targets[i]} Kenyon" for i in top))
    groups = {
        "apl": apl,
        "alpn": np.where(klass == "ALPN")[0],
        "inhib": np.where(inhib)[0],
        "left": np.where(side == "left")[0],
        "optic": np.where(np.isin(sup, ["optic", "visual_projection", "visual_centrifugal"]))[0],
        "half": rng.choice(N, N // 2, replace=False),
    }
    for k, g in groups.items():
        log(f"  lesión {k:6}: {len(g):,} neuronas")

    # ---------- entrada ----------
    m = max(8, len(inputs) // 8)
    win = np.stack([rng.choice(inputs, size=m, replace=False) for _ in range(V)])
    Win = torch.zeros(V, N, device=DEV)
    for c in range(V):
        Win[c, torch.from_numpy(win[c]).to(DEV)] = a.gain
    G = Graph(W)
    KCt = torch.from_numpy(KC).to(DEV)

    def step(X, ids, mask=None):
        U = Win[ids].T
        for _ in range(a.ticks):
            X = torch.tanh(G @ X + U)
            if mask is not None:
                X = X * mask
        return X

    def states(seqs, mask=None):
        L = min(len(q) for q in seqs)
        M = torch.from_numpy(np.stack([q[:L] for q in seqs], 1)).to(DEV)
        X = torch.zeros(N, len(seqs), device=DEV)
        out = np.zeros((L, len(seqs), len(KC)), np.float32)
        for t in range(L):
            X = step(X, M[t], mask)
            out[t] = X[KCt].T.cpu().numpy()
        return out

    # ---------- textos ----------
    for b in books:
        t = b["text"]
        b["test_ids"] = enc(t[-a.test - 200:])
        tr = t[: len(t) - a.test - 200]
        b["train_ids"] = enc((tr * math.ceil((a.train + 1) / len(tr)))[: a.train + 1])
        b["words"] = set(w for w in re.findall(r"[^\W\d_]+", tr) if len(w) >= 2)

    # ---------- lectura + aprendizaje ----------
    heads = [Readout(len(KC), V, a.lr, a.wd) for _ in books]
    SEQ = torch.from_numpy(np.stack([b["train_ids"] for b in books], 1)).to(DEV)
    X = torch.zeros(N, nb, device=DEV)
    warmX = []
    norm = None
    buf = [[] for _ in books]
    bufy = [[] for _ in books]
    Z = lambda k, S: np.clip((S - norm[k][0]) / norm[k][1], -5, 5)  # noqa: E731
    t0 = time.time()
    log(f"leyendo {a.train:,} letras × {nb} libros…")
    for t in range(a.train):
        X = step(X, SEQ[t])
        kc = X[KCt].T.cpu().numpy()
        if norm is None:
            warmX.append(kc)
            if len(warmX) == a.warm:
                S = np.stack(warmX)
                norm = []
                for k in range(nb):
                    mu, sd = S[:, k].mean(0), S[:, k].std(0)
                    norm.append((mu, np.maximum(sd, np.median(sd) + 1e-9)))
                for k in range(nb):
                    y = books[k]["train_ids"][1:a.warm + 1]
                    for i in range(0, a.warm, a.batch):
                        heads[k].step(Z(k, S[i:i + a.batch, k]), y[i:i + a.batch])
                warmX = None
            continue
        for k in range(nb):
            buf[k].append(kc[k])
            bufy[k].append(books[k]["train_ids"][t + 1])
        if len(buf[0]) >= a.batch:
            for k in range(nb):
                heads[k].step(Z(k, np.asarray(buf[k])), np.asarray(bufy[k]))
                buf[k].clear()
                bufy[k].clear()
        if (t + 1) % 3000 == 0:
            el = time.time() - t0
            log(f"  {t + 1:,} letras · {el / 60:.1f} min · faltan ~{el / (t + 1) * (a.train - t - 1) / 60:.0f} min")

    np.savez(ROOT / "logs/live_heads.npz", W=np.stack([h.W for h in heads]), mu=np.stack([n[0] for n in norm]), sd=np.stack([n[1] for n in norm]))

    # ---------- exámenes: sano y con cada lesión ----------
    def generate(k, prefix, mask=None, n=240, temp=0.8, seed=1):
        r = np.random.default_rng(seed)
        X = torch.zeros(N, 1, device=DEV)
        for c in enc(prefix):
            X = step(X, torch.tensor([c], device=DEV), mask)
        out = ""
        for _ in range(n):
            p = heads[k].probs(Z(k, X[KCt, 0].cpu().numpy()[None]))[0] ** (1 / temp)
            c = int(r.choice(V, p=p / p.sum()))
            out += vocab[c]
            X = step(X, torch.tensor([c], device=DEV), mask)
        words = [w for w in re.findall(r"[^\W\d_]+", out) if len(w) >= 2]
        return out, sum(w in books[k]["words"] for w in words) / max(1, len(words))

    results = {}
    for name in ["none"] + list(groups):
        mask = None
        if name != "none":
            mask = torch.ones(N, 1, device=DEV)
            mask[torch.from_numpy(groups[name]).to(DEV)] = 0
        S = states([b["test_ids"] for b in books], mask)
        bpc = [float(heads[k].surprise(Z(k, S[200:-1, k]), books[k]["test_ids"][201:len(S)]).mean()) for k in range(nb)]
        sample, words = generate(0, FAMOUS["shakespeare"][0], mask)
        kc_active = float((np.abs(S[200:, 0]) > 0.05).mean())
        results[name] = {"bpc": [round(v, 4) for v in bpc], "sample": sample, "words": round(words, 3), "kcActive": round(kc_active, 4)}
        log(f"  {name:6} → " + " · ".join(f"{b['id']}={v:.2f}" for b, v in zip(books, bpc)) + f" · palabras {words:.0%} · Kenyon activas {kc_active:.0%} · «{sample[:60]}»")

    # ---------- referencia para verificar el motor del navegador ----------
    ref_ids = enc("to be, or not to be")
    X = torch.zeros(N, 1, device=DEV)
    for c in ref_ids:
        X = step(X, torch.tensor([c], device=DEV))
    ref_kc = X[KCt[:64], 0].cpu().numpy()
    ref_p = heads[0].probs(Z(0, X[KCt, 0].cpu().numpy()[None]))[0]

    # ---------- exportar ----------
    pos = C["positions"][keep].astype(np.float64)
    pos -= pos.mean(0)
    pos /= np.abs(pos).max()
    flags = np.zeros(N, np.uint8)
    flags[inputs] |= 1
    flags[KC] |= 2
    for bit, k in enumerate(groups, start=2):
        flags[groups[k]] |= 1 << bit

    cols = W.indices.astype(np.int64)
    delta = cols.copy()
    for i in range(N):
        s, e = W.indptr[i], W.indptr[i + 1]
        if e - s > 1:
            delta[s + 1:e] = cols[s + 1:e] - cols[s:e - 1]

    arrays = {
        "rowptr": W.indptr.astype(np.uint32),
        "cols": delta.astype(np.uint32),
        "cnt": cnt,
        "win": win.astype(np.uint32).ravel(),
        "kc": KC.astype(np.uint32),
        "pos": np.round(pos * 32767).astype(np.int16).ravel(),
        "flags": flags,
    }
    for k, b in enumerate(books):
        Wh = heads[k].W
        sc = np.abs(Wh).max(0) / 32767 + 1e-12
        arrays[f"w_{b['id']}"] = np.round(Wh / sc).astype(np.int16).ravel()
        arrays[f"s_{b['id']}"] = sc.astype(np.float32)
        arrays[f"mu_{b['id']}"] = norm[k][0].astype(np.float32)
        arrays[f"sd_{b['id']}"] = norm[k][1].astype(np.float32)
    header = {"N": N, "nnz": int(W.nnz), "V": V, "vocab": "".join(vocab), "ticks": a.ticks, "gain": a.gain, "scale": a.scale,
              "books": [b["id"] for b in books], "groups": list(groups), "inputsPerLetter": m, "nKC": len(KC),
              "ref": {"text": "to be, or not to be", "kc": [float(v) for v in ref_kc], "p": [float(v) for v in ref_p]},
              "arrays": {}}
    blob = bytearray()
    for k, arr in arrays.items():
        while len(blob) % 8:
            blob.append(0)
        header["arrays"][k] = [len(blob), arr.dtype.str.lstrip("<|"), int(arr.size)]
        blob += arr.tobytes()
    hj = json.dumps(header).encode()
    out = ROOT / "web/public/lab/brain.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    pad = (8 - (4 + len(hj)) % 8) % 8
    raw = struct.pack("<I", len(hj) + pad) + hj + b" " * pad + bytes(blob)
    out.write_bytes(gzip.compress(raw, 9))
    log(f"exportado {out} · {len(raw) / 1e6:.1f} MB → {out.stat().st_size / 1e6:.1f} MB comprimido")

    lab = {
        "neurons": N, "connections": int(W.nnz), "inputs": int(len(inputs)), "kenyon": int(len(KC)), "train": a.train,
        "books": [{"id": b["id"], "title": b["title"]} for b in books],
        "groups": {k: int(len(v)) for k, v in groups.items()},
        "aplTargets": [int(kc_targets[i]) for i in apl],
        "results": results, "bpcRandom": round(math.log2(V), 4),
    }
    (ROOT / "web/public/data/lab.json").write_text(json.dumps(lab, ensure_ascii=False, indent=1))
    log(f"listo en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
