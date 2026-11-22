"""Compara ritmos de aprendizaje sobre los mismos estados del cerebro real (CPU, en paralelo a la corrida)."""
import sys, time, math
import numpy as np, scipy.sparse as sp
sys.path.insert(0, "pipeline"); np.seterr(all="ignore")
from books import load_all
from connectome import load_connectome
from run import Readout
rng = np.random.default_rng(42)
books = load_all(); vocab = sorted(set("".join(b["text"] for b in books))); idx = {c: i for i, c in enumerate(vocab)}; V = len(vocab)
b = books[0]; t = b["text"]; tr = t[: len(t) - 4000 - 200]
ids = np.array([idx[c] for c in tr[:8001]], np.int32)
# mismas llamadas al generador que run.py, para reproducir exactamente la entrada
C = load_connectome(rng=rng); N = C["W"].shape[0]
W = (C["W"] * 0.01).astype(np.float32).tocsr()
coo = W.tocoo(); _ = rng.permutation(coo.row)
inputs, KC = C["inputs"], C["outputs_pref"]
Win = np.zeros((V, N), np.float32)
for c in range(V):
    sel = rng.choice(inputs, size=max(8, len(inputs) // 8), replace=False); Win[c, sel] = 3.0
D = np.load(sorted(__import__("glob").glob("logs/cache_*.npz"))[0], allow_pickle=True)
norm = D["norm"].item()["shakespeare"]; Xt, yt = D["testZ"].item()["shakespeare"]
Xt = np.clip((Xt - norm[0]) / norm[1], -5, 5)  # el examen también normalizado (antes faltaba)
import os
if os.path.exists("logs/lrtest_states.npy"):
    S = np.load("logs/lrtest_states.npy")
else:
    t0 = time.time(); x = np.zeros(N, np.float32); S = np.zeros((8000, len(KC)), np.float32)
    for i in range(8000):
        for _ in range(3): x = np.tanh(W @ x + Win[ids[i]])
        S[i] = x[KC]
    np.save("logs/lrtest_states.npy", S)
    print(f"estados listos en {(time.time()-t0)/60:.1f} min", flush=True)
Z = np.clip((S - norm[0]) / norm[1], -5, 5)
class SGD:
    def __init__(self, d, lr, mom=0.9, wd=1e-4):
        self.W = np.zeros((d + 1, V), np.float32); self.v = np.zeros_like(self.W); self.lr, self.mom, self.wd = lr, mom, wd
    def probs(self, X):
        Zl = X @ self.W[:-1] + self.W[-1]; Zl -= Zl.max(1, keepdims=True); P = np.exp(Zl); return P / P.sum(1, keepdims=True)
    def step(self, X, y):
        P = self.probs(X); P[np.arange(len(y)), y] -= 1
        G = np.vstack([X.T @ P, P.sum(0, keepdims=True)]) / len(y) + self.wd * self.W
        self.v = self.mom * self.v - self.lr * G; self.W += self.v
    def surprise(self, X, y):
        P = self.probs(X); return -np.log2(P[np.arange(len(y)), y] + 1e-12)
def curve(h, F, Ft):
    out = []
    for e in range(0, 8000, 128):
        h.step(F[e:e+128], ids[e+1:e+1+len(F[e:e+128])])
        if (e // 128) in (10, 30, 62): out.append(f"{e+128}: {h.surprise(Ft, yt).mean():.2f}")
    return " · ".join(out)
E = np.eye(V, dtype=np.float32)
# letra actual de cada estado de examen (el estado t corresponde a la letra yt-1)
Dt = __import__("numpy").load(sorted(__import__("glob").glob("logs/cache_*.npz"))[0], allow_pickle=True)
test_ids = np.array([idx[c] for c in t[-4000 - 200:]], np.int32)
prev = test_ids[200:len(test_ids) - 1][: len(yt)]
for lr, lb in [(0.0, 0.1), (2e-4, 0.03), (2e-4, 0.1), (1e-3, 0.1), (3e-3, 0.1)]:
    print(f"Kenyon Adam lr={lr} base={lb}: " + curve(Readout(len(KC), V, lr, 1e-4, lb), Z, Xt), flush=True)
for lr in (0.01, 0.03, 0.1):
    print(f"Kenyon SGD lr={lr}: " + curve(SGD(len(KC), lr), Z, Xt), flush=True)
print("techo SGD 0.3: " + curve(SGD(V, 0.3), E[ids[:8000]], E[prev]), flush=True)
