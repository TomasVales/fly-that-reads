"""Carga el conectoma de FlyWire (FAFB v783) desde data/, o arma una red sintética del mismo tamaño."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent.parent / "data"
INHIBITORY = {"GABA", "GLUT"}  # en la mosca el glutamato suele inhibir (receptores GluCl)


def _find(*words):
    for f in sorted(DATA.glob("*")):
        n = f.name.lower()
        if all(w in n for w in words):
            return f
    return None


def _read(f):
    return pd.read_csv(f, compression="gzip" if f.suffix == ".gz" else None, low_memory=False)


def load_connectome(synthetic=False, rng=None):
    rng = rng or np.random.default_rng(0)
    if synthetic or _find("connect") is None:
        if not synthetic:
            print("⚠️  no encontré los archivos de FlyWire en data/ → uso una red sintética", flush=True)
        return _synthetic(rng)
    return _flywire(rng)


def _flywire(rng):
    fc = _find("connect")
    con = _read(fc)
    cols = {c.lower(): c for c in con.columns}
    pre = cols.get("pre_root_id") or next(c for k, c in cols.items() if "pre" in k)
    post = cols.get("post_root_id") or next(c for k, c in cols.items() if "post" in k)
    cnt = cols.get("syn_count") or next((c for k, c in cols.items() if "syn" in k or "count" in k), None)
    ntc = cols.get("nt_type") or next((c for k, c in cols.items() if "nt" in k and "type" in k), None)
    print(f"conexiones: {fc.name} · columnas {list(con.columns)}", flush=True)

    # neurotransmisor por neurona (si las conexiones no lo traen)
    nt_by_neuron = None
    fnt = _find("neurotransmitter") or _find("nt")
    if fnt is not None:
        nt = _read(fnt)
        nc = {c.lower(): c for c in nt.columns}
        rid = nc.get("root_id") or next(c for k, c in nc.items() if "root" in k)
        ntt = nc.get("nt_type") or nc.get("predicted_nt") or next((c for k, c in nc.items() if "nt" in k), None)
        if ntt:
            nt_by_neuron = dict(zip(nt[rid].astype("int64"), nt[ntt].astype(str).str.upper()))

    ids = pd.unique(pd.concat([con[pre], con[post]]).astype("int64"))
    pos_of = {r: i for i, r in enumerate(ids)}
    N = len(ids)
    pi = con[pre].astype("int64").map(pos_of).to_numpy()
    qi = con[post].astype("int64").map(pos_of).to_numpy()
    w = con[cnt].to_numpy(np.float32) if cnt else np.ones(len(con), np.float32)
    if ntc:
        sign = np.where(con[ntc].astype(str).str.upper().isin(INHIBITORY), -1.0, 1.0)
    elif nt_by_neuron:
        sign = np.array([-1.0 if nt_by_neuron.get(r, "ACH") in INHIBITORY else 1.0 for r in con[pre].astype("int64")])
    else:
        sign = np.ones(len(con))
    synapses = int(w.sum())
    # W[post, pre]: la neurona post recibe de pre. Primero se suman las sinapsis de cada par (vienen separadas
    # por neuropilo) y después se usa log(1+sinapsis) para que unas pocas conexiones gigantes no dominen
    W = sp.csr_matrix(((w * sign).astype(np.float32), (qi, pi)), shape=(N, N))
    W.sum_duplicates()
    W.data = (np.sign(W.data) * np.log1p(np.abs(W.data))).astype(np.float32)
    W.eliminate_zeros()

    # clases: sensoriales (entrada) y descendentes (salida hacia el cuerpo)
    inputs = outputs = np.array([], int)
    fcl = _find("classif") or _find("hierarch")
    if fcl is not None:
        cl = _read(fcl)
        cc = {c.lower(): c for c in cl.columns}
        rid = cc.get("root_id") or next(c for k, c in cc.items() if "root" in k)
        sup = cc.get("super_class") or next(c for k, c in cc.items() if "super" in k)
        m = dict(zip(cl[rid].astype("int64"), cl[sup].astype(str).str.lower()))
        klass = np.array([m.get(r, "") for r in ids])
        sub = cl[cc.get("class", sup)].astype(str) if "class" in cc else None
        mclass = dict(zip(cl[rid].astype("int64"), cl[cc["class"]].astype(str))) if "class" in cc else {}
        ktype = np.array([mclass.get(r, "") for r in ids])
        # entrada: neuronas olfativas (el camino principal hacia el centro de memoria)
        inputs = np.where(ktype == "olfactory")[0]
        # salida: células de Kenyon del cuerpo fungiforme (donde la mosca aprende)
        outputs = np.where(ktype == "Kenyon_Cell")[0]
        print(f"clases: {pd.Series(klass).value_counts().head(12).to_dict()}", flush=True)
    if len(inputs) < 200:
        inputs = rng.choice(N, 5000, replace=False)

    # posiciones 3D
    positions = np.full((N, 3), np.nan, np.float32)
    fco = _find("coord")
    if fco is not None:
        co = _read(fco)
        oc = {c.lower(): c for c in co.columns}
        rid = oc.get("root_id") or next(c for k, c in oc.items() if "root" in k)
        if "position" in oc:
            xyz = co[oc["position"]].astype(str).str.replace(r"[\[\]()]", "", regex=True).str.strip().str.split(r"[\s,]+", expand=True).iloc[:, :3].astype(float)
        else:
            xyz = co[[oc.get("x"), oc.get("y"), oc.get("z")]].astype(float)
        xyz.index = co[rid].astype("int64")
        xyz = xyz[~xyz.index.duplicated()]
        common = [r for r in ids if r in xyz.index]
        P = xyz.loc[common].to_numpy(np.float32)
        positions[[pos_of[r] for r in common]] = P
        # FlyWire: y crece hacia abajo → la invierto para que el cerebro quede derecho
        positions[:, 1] *= -1
    have = np.where(~np.isnan(positions[:, 0]))[0]
    viz = rng.permutation(have) if len(have) else rng.permutation(N)
    if not len(have):
        positions = _brain_cloud(N, rng)

    return {"W": W, "inputs": inputs, "outputs_pref": outputs, "viz": viz, "positions": np.nan_to_num(positions),
            "synapses": synapses, "source": "FlyWire FAFB v783", "ids": np.asarray(ids, np.int64)}


def _brain_cloud(n, rng):
    part = rng.random(n)
    p = np.zeros((n, 3), np.float32)
    c = part < 0.45
    p[c] = rng.normal(0, [0.25, 0.3, 0.28], (c.sum(), 3))
    for s, m in ((-1, (part >= 0.45) & (part < 0.72)), (1, part >= 0.72)):
        k = m.sum()
        p[m] = np.stack([s * (0.65 + rng.normal(0, 0.13, k)), rng.normal(0, 0.32, k), rng.normal(0, 0.2, k)], 1)
    return p


def _synthetic(rng, N=139_255, nnz=3_732_460):
    # grados con cola larga, como en un cerebro real
    out_deg = rng.pareto(1.6, N) + 1
    out_deg = (out_deg / out_deg.sum() * nnz).astype(int) + 1
    pre = np.repeat(np.arange(N), out_deg)[:nnz]
    post = rng.integers(0, N, len(pre))
    w = np.log1p(rng.pareto(1.3, len(pre)) * 3 + 1).astype(np.float32)
    sign = np.where(rng.random(len(pre)) < 0.3, -1.0, 1.0).astype(np.float32)
    W = sp.csr_matrix((w * sign, (post, pre)), shape=(N, N))
    W.sum_duplicates()
    return {"W": W, "inputs": rng.choice(N, 5000, replace=False), "outputs_pref": rng.choice(N, 1300, replace=False),
            "viz": rng.permutation(N), "positions": _brain_cloud(N, rng), "synapses": int(nnz * 14), "source": "synthetic"}
