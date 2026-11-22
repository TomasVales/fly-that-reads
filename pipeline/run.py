"""
La mosca que lee — experimento con el cerebro real (FlyWire FAFB v783).

Modelo (todo sobre el cableado real, sin inventar conexiones):
  - Cada sinapsis real es una conexión; su peso crece con el número de sinapsis (log) y su signo sale del
    neurotransmisor (GABA y glutamato inhiben, el resto excita).
  - Las letras entran por las neuronas olfativas (el camino principal al centro de aprendizaje de la mosca).
  - Cada letra se "huele" durante varios pasos de tiempo; la actividad viaja capa por capa por el cerebro, así que
    las capas profundas guardan un recuerdo de las letras anteriores.
  - La mosca aprende solo en las conexiones que salen de sus células de Kenyon (cuerpo fungiforme), como en la
    mosca real. El cableado no cambia.
  Control: el mismo cerebro con las conexiones mezcladas al azar, leyendo Shakespeare.

Exámenes en cada punto de control:
  - sorpresa por letra (bits) en texto que nunca leyó
  - qué escribe (y qué parte son palabras reales del libro)
  - si reconoce a su autor frente a los otros tres
  - cuántos intentos le faltan para completar la frase famosa

Uso: python pipeline/run.py [--train 160000] [--out web/public/data/run.json]
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parent))
from books import load_all  # noqa: E402
from connectome import load_connectome  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# numpy + Accelerate (macOS) tira avisos falsos de overflow en matmul float32; verificamos que todo sea finito
np.seterr(all="ignore")

FAMOUS = {
    "shakespeare": ("to be, or not to be, ", "that is the question"),
    "quijote": ("en un lugar de la mancha, de cuyo ", "nombre no quiero acordarme"),
    "fierro": ("aquí me pongo a cantar al compás ", "de la vigüela"),
    "dante": ("nel mezzo del cammin di ", "nostra vita"),
}


DEV = "mps" if torch.backends.mps.is_available() else "cpu"


class Graph:
    """multiplicación por el cableado en la GPU (suma dispersa: cada sinapsis aporta peso × actividad)"""

    def __init__(self, Wcsr):
        c = Wcsr.tocoo()
        self.rows = torch.from_numpy(c.row.astype(np.int64)).to(DEV)
        self.cols = torch.from_numpy(c.col.astype(np.int64)).to(DEV)
        self.vals = torch.from_numpy(c.data.astype(np.float32)).to(DEV)
        self.n = Wcsr.shape[0]

    def __matmul__(self, X):
        y = torch.zeros_like(X)
        y.index_add_(0, self.rows, X[self.cols] * self.vals[:, None])
        return y


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Readout:
    """softmax lineal (conexiones de salida de las células de Kenyon), entrenado online con Adam"""

    def __init__(self, d, v, lr, wd=0.0, lr_bias=0.1):
        self.W = np.zeros((d + 1, v), np.float32)
        # la "intuición base" (qué letras son comunes) aprende más rápido que las conexiones de las Kenyon
        self.rate = np.full((d + 1, 1), lr, np.float32)
        self.rate[-1] = lr_bias
        self.m = np.zeros_like(self.W)
        self.s = np.zeros_like(self.W)
        self.t, self.lr, self.wd = 0, lr, wd

    def probs(self, X):
        Z = X @ self.W[:-1] + self.W[-1]
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(1, keepdims=True)

    def step(self, X, y):
        P = self.probs(X)
        P[np.arange(len(y)), y] -= 1
        G = np.vstack([X.T @ P, P.sum(0, keepdims=True)]) / len(y)
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * G
        self.s = 0.999 * self.s + 0.001 * G * G
        mh = self.m / (1 - 0.9 ** self.t)
        sh = self.s / (1 - 0.999 ** self.t)
        self.W -= self.rate * (mh / (np.sqrt(sh) + 1e-8) + self.wd * self.W)

    def surprise(self, X, y):
        """bits por letra de cada posición"""
        P = self.probs(X)
        assert np.isfinite(P).all(), "probabilidades inválidas"
        return -np.log2(P[np.arange(len(y)), y] + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=160_000, help="letras que lee cada mosca")
    ap.add_argument("--test", type=int, default=4000)
    ap.add_argument("--scale", type=float, default=0.01, help="peso de cada conexión (× log(1+sinapsis))")
    ap.add_argument("--ticks", type=int, default=3, help="pasos de tiempo por letra")
    ap.add_argument("--alpha", type=float, default=1.0, help="1 = sin fuga; <1 = las neuronas retienen actividad")
    ap.add_argument("--gain", type=float, default=3.0, help="fuerza del estímulo olfativo")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warm", type=int, default=3000, help="letras para calibrar la escala de las Kenyon")
    ap.add_argument("--checkpoints", type=int, default=40)
    ap.add_argument("--viz", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bench", type=int, default=0, help="solo medir velocidad con N letras")
    ap.add_argument("--out", default=str(ROOT / "web/public/data/run.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # ---------- libros ----------
    books = load_all()
    vocab = sorted(set("".join(b["text"] for b in books)))
    idx = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    enc = lambda s: np.array([idx[c] for c in s if c in idx], np.int32)  # noqa: E731
    log(f"alfabeto: {V} símbolos → teclear al azar = {math.log2(V):.2f} bits/letra")
    for b in books:
        t = b["text"]
        b["test_ids"] = enc(t[-a.test - 200:])
        tr = t[: len(t) - a.test - 200]
        b["reads"] = a.train / len(tr)
        b["train_ids"] = enc((tr * math.ceil(a.train / len(tr)))[: a.train + 1])
        b["words"] = set(w for w in re.findall(r"[^\W\d_]+", tr) if len(w) >= 2)
        # largo de la obra a escribir "de cero": Hamlet para Shakespeare, la obra entera para el resto
        if b["id"] == "shakespeare":
            i0 = t.find("the tragedy of hamlet, prince of denmark")
            i1 = t.find("the first part of king henry the fourth", i0) if i0 >= 0 else -1
            b["work"], b["work_len"] = "Hamlet", (i1 - i0 if 0 <= i0 < i1 else 175_000)
        else:
            b["work"], b["work_len"] = b["title"], len(t)
        log(f"{b['id']:12} lee {a.train:,} letras (el libro {b['reads']:.2f} veces) · obra a escribir: {b['work']} ({b['work_len']:,} letras)")

    # ---------- cerebro ----------
    C = load_connectome(rng=rng)
    N = C["W"].shape[0]
    W = (C["W"] * a.scale).astype(np.float32).tocsr()
    coo = W.tocoo()
    Wrand = sp.csr_matrix((coo.data, (rng.permutation(coo.row), coo.col)), shape=W.shape).astype(np.float32)
    inputs, KC = C["inputs"], C["outputs_pref"]
    log(f"cerebro: {N:,} neuronas · {W.nnz:,} conexiones · {C['synapses']:,} sinapsis · {len(inputs):,} olfativas · {len(KC):,} Kenyon")
    Win_np = np.zeros((V, N), np.float32)
    for c in range(V):
        sel = rng.choice(inputs, size=max(8, len(inputs) // 8), replace=False)
        Win_np[c, sel] = a.gain
    Win = torch.from_numpy(Win_np).to(DEV)
    G, Grand = Graph(W), Graph(Wrand)
    KCt = torch.from_numpy(KC.astype(np.int64)).to(DEV)
    # neuronas a dibujar: todas las del camino de aprendizaje + una muestra del resto
    path = np.concatenate([inputs, KC]).astype(int)
    rest = np.setdiff1d(C["viz"], path)
    viz = np.concatenate([path, rng.choice(rest, size=max(0, a.viz - len(path)), replace=False)])
    cols = [b["id"] for b in books] + ["random"]
    K = len(cols)
    nb = len(books)
    log(f"motor: {DEV}")

    def step(g, X, ids):
        """una letra: se 'huele' durante varios pasos; X (N, k), ids (k,)"""
        U = Win[ids].T
        for _ in range(a.ticks):
            X = torch.tanh(g @ X + U) if a.alpha == 1.0 else (1 - a.alpha) * X + a.alpha * torch.tanh(g @ X + U)
        return X

    def states(g, seqs_):
        """corre varias secuencias del mismo largo a la vez; devuelve estados de Kenyon (T, k, nKC)"""
        L = min(len(q) for q in seqs_)
        M = torch.from_numpy(np.stack([q[:L] for q in seqs_], 1).astype(np.int64)).to(DEV)
        X = torch.zeros(N, len(seqs_), device=DEV)
        out = np.zeros((L, len(seqs_), len(KC)), np.float32)
        for t in range(L):
            X = step(g, X, M[t])
            out[t] = X[KCt].T.cpu().numpy()
        return out

    cache = ROOT / "logs" / f"cache_s{a.seed}_sc{a.scale}_t{a.ticks}_a{a.alpha}_g{a.gain}_w{a.warm}_te{a.test}.npz"
    if cache.exists():
        log(f"usando calibración y exámenes guardados ({cache.name})")
        D = np.load(cache, allow_pickle=True)
        norm = D["norm"].item()
        testZ = D["testZ"].item()
        lines = D["lines"].item()
    else:
        # ---------- calibración (escala de cada Kenyon) ----------
        log("calibrando…")
        S = states(G, [b["train_ids"][: a.warm] for b in books])
        norm = {}
        for k, b in enumerate(books):
            mu, sd = S[:, k].mean(0), S[:, k].std(0)
            norm[b["id"]] = (mu, np.maximum(sd, np.median(sd) + 1e-9))
        S = states(Grand, [books[0]["train_ids"][: a.warm]])
        mu, sd = S[:, 0].mean(0), S[:, 0].std(0)
        norm["random"] = (mu, np.maximum(sd, np.median(sd) + 1e-9))
        Z = lambda col, S: np.clip((S - norm[col][0]) / norm[col][1], -5, 5)  # noqa: E731

        # ---------- exámenes fijos (no dependen de lo aprendido) ----------
        log("preparando exámenes…")
        testZ = {}
        S = states(G, [b["test_ids"] for b in books])
        for k, b in enumerate(books):
            testZ[b["id"]] = (S[200:-1, k], b["test_ids"][201:len(S)])
        S = states(Grand, [books[0]["test_ids"]])
        testZ["random"] = (S[200:-1, 0], books[0]["test_ids"][201:len(S)])
        lines = {}
        for b in books:
            pre, comp = FAMOUS[b["id"]]
            ids = enc(pre + comp)
            S = states(G, [ids])
            lines[b["id"]] = (S[len(pre) - 1: -1, 0], ids[len(pre):])
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, norm=np.array(norm, dtype=object), testZ=np.array(testZ, dtype=object), lines=np.array(lines, dtype=object))
    Z = lambda col, S: np.clip((S - norm[col][0]) / norm[col][1], -5, 5)  # noqa: E731

    heads = {col: Readout(len(KC), V, a.lr, a.wd) for col in cols}
    seqs = [b["train_ids"] for b in books] + [books[0]["train_ids"]]
    vizt = torch.from_numpy(viz.astype(np.int64)).to(DEV)
    Xviz = np.zeros((len(viz), nb), np.float32)
    marks = sorted({max(1, int(a.train * (i / a.checkpoints) ** 1.8)) for i in range(a.checkpoints + 1)})
    ck = {col: [] for col in cols}
    bufX = {col: [] for col in cols}
    bufY = {col: [] for col in cols}
    prime = {b["id"]: b["text"][-a.test - 400: -a.test - 340] for b in books}
    CH = 100  # largo de cada fragmento para reconocer al autor

    def generate_all(n=200, temp=0.8):
        """los 4 libros escriben a la vez, cada uno desde un comienzo de su propio libro"""
        X = torch.zeros(N, nb, device=DEV)
        pr = [enc(prime[b["id"]]) for b in books]
        L = min(len(p) for p in pr)
        for t in range(L):
            X = step(G, X, torch.tensor([p[t] for p in pr], device=DEV))
        outs = ["" for _ in books]
        r = np.random.default_rng(len(ck[cols[0]]) + 7)
        for _ in range(n):
            kc = X[KCt].T.cpu().numpy()
            nxt = []
            for k, b in enumerate(books):
                p = heads[b["id"]].probs(Z(b["id"], kc[k][None]))[0] ** (1 / temp)
                c = int(r.choice(V, p=p / p.sum()))
                outs[k] += vocab[c]
                nxt.append(c)
            X = step(G, X, torch.tensor(nxt, device=DEV))
        res = {}
        for k, b in enumerate(books):
            words = [w for w in re.findall(r"[^\W\d_]+", outs[k]) if len(w) >= 2]
            res[b["id"]] = (outs[k], sum(w in b["words"] for w in words) / max(1, len(words)))
        return res

    def recognize(col):
        """¿distingue un fragmento de su autor de uno de otro autor? (azar = 50%)"""
        own = [heads[col].surprise(Z(col, testZ[col][0][i:i + CH]), testZ[col][1][i:i + CH]).mean() for i in range(0, len(testZ[col][1]) - CH, CH)]
        wins = tot = 0
        for o in [c for c in cols[:-1] if c != col]:
            other = [heads[col].surprise(Z(col, testZ[o][0][i:i + CH]), testZ[o][1][i:i + CH]).mean() for i in range(0, len(testZ[o][1]) - CH, CH)]
            for s1, s2 in zip(own, other):
                wins += s1 < s2
                tot += 1
        return wins / max(1, tot)

    def save(done):
        pos = C["positions"][viz].astype(np.float64)
        pos = pos - pos.mean(0)
        pos = pos / np.abs(pos).max()
        run = {
            "meta": {
                "demo": False, "done": done, "progress": round(ck[cols[0]][-1]["chars"] / a.train, 4),
                "neurons": int(N), "synapses": int(C["synapses"]), "connections": int(W.nnz),
                "inputs": int(len(inputs)), "kenyon": int(len(KC)),
                "bpcRandom": round(math.log2(V), 4), "bpcHuman": 1.0, "target": 50, "charsPerSecond": 15,
                "source": C["source"], "vocab": V,
                "params": {k: getattr(a, k) for k in ["train", "test", "scale", "ticks", "alpha", "gain", "lr", "wd", "batch"]},
                "groups": {"inputs": [0, len(inputs)], "kenyon": [len(inputs), len(inputs) + len(KC)]},
            },
            "brain": {"positions": [round(float(v), 4) for v in pos.reshape(-1)]},
            "books": [
                {"id": b["id"], "title": b["title"], "author": b["author"], "lang": b["lang"], "excerpt": b["excerpt"],
                 "reads": round(b["reads"], 2), "work": b["work"], "workLen": int(b["work_len"]),
                 "famous": {"prefix": FAMOUS[b["id"]][0], "completion": FAMOUS[b["id"]][1]},
                 "checkpoints": ck[b["id"]]}
                for b in books
            ],
            "controls": [{"id": "random", "label": "Shakespeare · random wiring", "checkpoints": ck["random"]}],
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(a.out).with_suffix(".tmp")
        tmp.write_text(json.dumps(run, ensure_ascii=False))
        tmp.replace(a.out)

    def checkpoint(t):
        gen = generate_all()
        for k, col in enumerate(cols):
            Xt, yt = testZ[col]
            e = {"chars": int(t), "bpc": round(float(heads[col].surprise(Z(col, Xt), yt).mean()), 4)}
            if col != "random":
                text, real = gen[col]
                lx, ly = lines[col]
                e.update({
                    "sample": text, "words": round(real, 3), "recog": round(recognize(col), 3),
                    "lineBits": round(float(heads[col].surprise(Z(col, lx), ly).sum()), 2),
                    "active": np.argsort(-np.abs(Xviz[:, k]))[:260].astype(int).tolist(),
                })
            ck[col].append(e)
        log(f"{t:>9,} letras · " + " · ".join(f"{c}={ck[c][-1]['bpc']:.2f}" for c in cols))
        b0 = ck[cols[0]][-1]
        log(f"          shakespeare: palabras reales {b0['words']:.0%} · reconoce autor {b0['recog']:.0%} · frase {b0['lineBits']:.0f} bits · «{b0['sample'][:70]}»")
        save(done=t >= a.train)

    log(f"leyendo… ({a.train:,} letras × {K} cerebros, {a.ticks} pasos por letra)")
    t0 = time.time()
    SEQ = torch.from_numpy(np.stack(seqs, 1).astype(np.int64)).to(DEV)
    if a.bench:
        tb = time.time()
        for t in range(a.bench):
            Xb = step(G, torch.zeros(N, nb, device=DEV) if t == 0 else Xb, SEQ[t, :nb])
            xr = step(Grand, torch.zeros(N, 1, device=DEV) if t == 0 else xr, SEQ[t, nb:])
            kc = Xb[KCt].T.cpu().numpy()
        dt = (time.time() - tb) / a.bench
        log(f"velocidad: {dt * 1000:.0f} ms por letra → {dt * a.train / 3600:.1f} h para {a.train:,} letras (sin exámenes)")
        return
    checkpoint(0)
    mi = 1
    Xb = torch.zeros(N, nb, device=DEV)
    xr = torch.zeros(N, 1, device=DEV)
    for t in range(a.train):
        Xb = step(G, Xb, SEQ[t, :nb])
        xr = step(Grand, xr, SEQ[t, nb:])
        kc = Xb[KCt].T.cpu().numpy()
        kr = xr[KCt, 0].cpu().numpy()
        for k in range(nb):
            bufX[cols[k]].append(kc[k])
            bufY[cols[k]].append(seqs[k][t + 1])
        bufX["random"].append(kr)
        bufY["random"].append(seqs[-1][t + 1])
        if (t + 1) % 500 == 0 or (mi < len(marks) and t + 2 >= marks[mi]):
            Xviz[:] = Xb[vizt].cpu().numpy()
        if len(bufX[cols[0]]) >= a.batch:
            for col in cols:
                heads[col].step(Z(col, np.asarray(bufX[col])), np.asarray(bufY[col]))
                bufX[col].clear()
                bufY[col].clear()
        if mi < len(marks) and t + 1 >= marks[mi]:
            checkpoint(t + 1)
            el = time.time() - t0
            log(f"   {100 * (t + 1) / a.train:.1f}% · {el / 60:.1f} min · faltan ~{el / (t + 1) * (a.train - t - 1) / 60:.0f} min")
            mi += 1
    log(f"listo → {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB) en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
