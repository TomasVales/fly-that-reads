import sys, numpy as np
sys.path.insert(0, "pipeline"); np.seterr(all="ignore")
import scipy.sparse as sp
from books import load_all
from connectome import load_connectome
from run import Readout
rng = np.random.default_rng(1)
bs = load_all(); b = bs[0]
vocab = sorted(set("".join(x["text"] for x in bs))); idx = {c: i for i, c in enumerate(vocab)}; V = len(vocab)
ids = np.array([idx[c] for c in b["text"][:12401]]); te = np.array([idx[c] for c in b["text"][-3000:]])
# 1) optimizador con la letra actual en one-hot → debería acercarse a ~3.3 (bigramas)
for lr, wd in [(3e-4, 1e-3), (3e-3, 1e-4), (1e-2, 0)]:
    h = Readout(V, V, lr, wd); E = np.eye(V, dtype=np.float32)
    for ep in range(3):
        for e in range(0, 12000, 128): h.step(E[ids[e:e+128]], ids[e+1:e+129])
    print(f"one-hot lr={lr} wd={wd}: {h.bpc(E[te[:-1]], te[1:]):.2f} bits", flush=True)
# 2) ¿llega la letra a las Kenyon? decodificar la letra ACTUAL desde las Kenyon
C = load_connectome(rng=rng); N = C["W"].shape[0]
W0 = C["W"].tocsr(); rs = np.asarray(np.abs(W0).sum(1)).ravel(); rs[rs == 0] = 1
inp, KC = C["inputs"], C["outputs_pref"]
for g, a in [(1.5, 0.3), (1.5, 1.0), (3.0, 1.0)]:
    W = (sp.diags(1 / rs) @ W0 * g).astype(np.float32).tocsr()
    Win = np.zeros((V, N), np.float32)
    for c in range(V):
        sel = rng.choice(inp, size=len(inp) // 8, replace=False); Win[c, sel] = 1.0
    x = np.zeros(N, np.float32); S = []
    for t in range(4000):
        x = (1 - a) * x + a * np.tanh(W @ x + Win[ids[t]]); S.append(x[KC].copy())
    S = np.array(S); mu, sd = S.mean(0), S.std(0); sd = np.maximum(sd, np.median(sd) + 1e-9); Z = np.clip((S - mu) / sd, -5, 5)
    h = Readout(len(KC), V, 3e-3, 1e-4)
    for ep in range(3):
        for e in range(0, 3000, 128): h.step(Z[e:e+128], ids[e:min(e+128,3000)])
    acc = (h.probs(Z[3000:]).argmax(1) == ids[3000:4000]).mean()
    act = (np.abs(S) > 1e-3).mean()
    print(f"gan={g} α={a}: letra actual adivinada {acc:.0%} (azar ~{1/V:.0%}, letra más común ~17%) · Kenyon activas {act:.0%}", flush=True)
