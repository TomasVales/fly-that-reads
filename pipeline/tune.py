"""Prueba rápida: ¿la actividad del cerebro real se propaga bien y la mosca aprende? (Shakespeare, pocas letras)"""
import sys, time, math
import numpy as np
sys.path.insert(0, "pipeline")
np.seterr(all="ignore")
from books import load_all
from connectome import load_connectome
from run import spectral_scale, Readout

rng = np.random.default_rng(1)
C = load_connectome(rng=rng)
N = C["W"].shape[0]
b = load_all()[0]
vocab = sorted(set("".join(x["text"] for x in load_all())))
idx = {c: i for i, c in enumerate(vocab)}
V = len(vocab)
ids = np.array([idx[c] for c in b["text"][:24000]], np.int32)
test = np.array([idx[c] for c in b["text"][-3200:]], np.int32)
inputs = C["inputs"]
R = C["outputs_pref"]
print("entradas", len(inputs), "salidas (Kenyon)", len(R))
TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 12000

import scipy.sparse as sp
W0 = C["W"].tocsr()
rowsum = np.asarray(np.abs(W0).sum(1)).ravel(); rowsum[rowsum == 0] = 1
Wn = sp.diags(1 / rowsum) @ W0  # cada neurona: fracción de sus sinapsis de entrada
for alpha, rho, ins, lr, wd in [(0.5, 1.0, 1.0, 3e-4, 1e-3), (0.5, 1.5, 1.0, 3e-4, 1e-3), (0.3, 1.5, 1.0, 3e-4, 1e-3), (0.3, 2.0, 2.0, 3e-4, 1e-3)]:
    W = (Wn * rho).astype(np.float32).tocsr()
    Win = np.zeros((V, N), np.float32)
    for c in range(V):
        sel = rng.choice(inputs, size=len(inputs) // 8, replace=False)
        Win[c, sel] = rng.choice([-1, 1], size=len(sel)) * ins
    def states(seq):
        x = np.zeros(N, np.float32); out = np.zeros((len(seq), len(R)), np.float32); act = []
        for t, c in enumerate(seq):
            x = (1 - alpha) * x + alpha * np.tanh(W @ x + Win[c]); out[t] = x[R]
            if t % 500 == 0: act.append(((np.abs(x) > 0.01).mean(), (np.abs(x) > 0.95).mean(), np.abs(x[R]).mean()))
        return out, act
    t0 = time.time()
    S, act = states(ids[: TRAIN + 1]); T, _ = states(test)
    mu = S[:2000].mean(0); sd = S[:2000].std(0); sd = np.maximum(sd, np.median(sd) + 1e-6)
    S = np.clip((S - mu) / sd, -5, 5); T = np.clip((T - mu) / sd, -5, 5)
    h = Readout(len(R), V, lr, wd)
    for e in range(0, TRAIN, 128):
        Xb = S[e:min(e + 128, TRAIN)]; h.step(Xb, ids[e + 1:e + 1 + len(Xb)])
    a = np.array(act)
    print(f"α={alpha} ganancia={rho} entrada={ins}: bits/letra {h.bpc(T[200:-1], test[201:]):.3f} · neuronas activas {a[:,0].mean():.0%} · saturadas {a[:,1].mean():.1%} · |x| salida {a[:,2].mean():.3f} · {time.time()-t0:.0f}s", flush=True)
