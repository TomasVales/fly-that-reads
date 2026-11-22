import sys, numpy as np, pandas as pd
sys.path.insert(0, "pipeline"); np.seterr(all="ignore")
import scipy.sparse as sp
from books import load_all
from connectome import load_connectome
from run import Readout
rng = np.random.default_rng(1)
bs = load_all(); b = bs[0]
vocab = sorted(set("".join(x["text"] for x in bs))); idx = {c: i for i, c in enumerate(vocab)}; V = len(vocab)
ids = np.array([idx[c] for c in b["text"][:5000]])
C = load_connectome(rng=rng); N = C["W"].shape[0]
cl = pd.read_csv("data/classification.csv.gz"); ids_all = pd.read_csv("data/connections_princeton.csv.gz", usecols=[0, 1])
allids = pd.unique(pd.concat([ids_all.iloc[:, 0], ids_all.iloc[:, 1]]).astype("int64")); pos = {r: i for i, r in enumerate(allids)}
cls = dict(zip(cl.root_id, cl["class"].astype(str)))
layer = {k: np.array([pos[r] for r, c in cls.items() if c == k and r in pos]) for k in ["olfactory", "ALPN", "Kenyon_Cell", "MBON"]}
W0 = C["W"].tocsr(); rs = np.asarray(np.abs(W0).sum(1)).ravel(); rs[rs == 0] = 1
def dec(S, y):
    mu, sd = S.mean(0), S.std(0); sd = np.maximum(sd, np.median(sd) + 1e-9); Z = np.clip((S - mu) / sd, -5, 5)
    h = Readout(S.shape[1], V, 1e-2, 0)
    for ep in range(4):
        for e in range(0, 3500, 128): h.step(Z[e:e+128], y[e:min(e+128, 3500)])
    return (h.probs(Z[3500:]).argmax(1) == y[3500:]).mean()
for name, Wm in [("normalizado g=1.5", (sp.diags(1/rs) @ W0 * 1.5)), ("sinapsis/100", W0 * 0.01)]:
    Wm = Wm.astype(np.float32).tocsr()
    Win = np.zeros((V, N), np.float32)
    for c in range(V):
        sel = rng.choice(layer["olfactory"], size=len(layer["olfactory"]) // 8, replace=False); Win[c, sel] = 3.0
    x = np.zeros(N, np.float32); rec = {k: [] for k in layer}
    for t in range(4500):
        x = np.tanh(Wm @ x + Win[ids[t]])
        for k in layer: rec[k].append(x[layer[k]].copy())
    for k in layer:
        S = np.array(rec[k])
        print(f"{name:18} {k:12} ({len(layer[k]):>5}) letra actual: {dec(S, ids[:4500]):.0%} · anterior: {dec(S[1:], ids[:4499]):.0%} · actividad media {np.abs(S).mean():.3f}", flush=True)
