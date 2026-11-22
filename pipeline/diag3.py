import sys, numpy as np, pandas as pd
sys.path.insert(0, "pipeline"); np.seterr(all="ignore")
import scipy.sparse as sp
from books import load_all
from connectome import load_connectome
from run import Readout
rng = np.random.default_rng(1)
bs = load_all(); b = bs[0]
vocab = sorted(set("".join(x["text"] for x in bs))); idx = {c: i for i, c in enumerate(vocab)}; V = len(vocab)
ids = np.array([idx[c] for c in b["text"][:20001]]); te = np.array([idx[c] for c in b["text"][-3000:]])
C = load_connectome(rng=rng); N = C["W"].shape[0]; inp, KC = C["inputs"], C["outputs_pref"]
W0 = C["W"].tocsr()
def z(S, mu=None, sd=None):
    if mu is None: mu, sd = S.mean(0), S.std(0); sd = np.maximum(sd, np.median(sd) + 1e-9)
    return np.clip((S - mu) / sd, -5, 5), mu, sd
for scale, ticks, a, gain in [(0.01, 3, 1.0, 3.0), (0.01, 3, 0.6, 3.0), (0.01, 4, 0.7, 3.0), (0.02, 3, 0.7, 3.0)]:
    Wm = (W0 * scale).astype(np.float32).tocsr()
    Win = np.zeros((V, N), np.float32)
    for c in range(V):
        sel = rng.choice(inp, size=len(inp) // 8, replace=False); Win[c, sel] = gain
    def run(seq):
        x = np.zeros(N, np.float32); out = np.zeros((len(seq), len(KC)), np.float32)
        for t, c in enumerate(seq):
            for _ in range(ticks): x = (1 - a) * x + a * np.tanh(Wm @ x + Win[c])
            out[t] = x[KC]
        return out
    S = run(ids); T = run(te)
    Z, mu, sd = z(S); ZT, _, _ = z(T, mu, sd)
    dec = []
    for back in range(4):
        h = Readout(len(KC), V, 1e-2, 0)
        for e in range(back, 6000, 128): h.step(Z[e:e+128], ids[e-back:e-back+len(Z[e:e+128])])
        dec.append((h.probs(Z[6000:8000]).argmax(1) == ids[6000-back:8000-back]).mean())
    best = None
    for lr, wd in [(3e-3, 1e-4), (1e-3, 1e-4)]:
        h = Readout(len(KC), V, lr, wd)
        for e in range(0, 20000, 128): h.step(Z[e:e+128], ids[e+1:e+1+len(Z[e:e+128])][:len(Z[e:e+128])])
        bp = h.bpc(ZT[200:-1], te[201:]); best = bp if best is None else min(best, bp)
    print(f"escala={scale} pasos={ticks} α={a}: Kenyon saben letra actual {dec[0]:.0%}, -1 {dec[1]:.0%}, -2 {dec[2]:.0%}, -3 {dec[3]:.0%} · examen {best:.2f} bits/letra", flush=True)
