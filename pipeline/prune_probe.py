import sys, numpy as np
sys.path.insert(0,'pipeline')
from connectome import load_connectome
C=load_connectome(rng=np.random.default_rng(42))
W=C['W'].tocsr(); N=W.shape[0]
inp, KC = C['inputs'], C['outputs_pref']
A=(W!=0).astype(np.float32).tocsr(); AT=A.T.tocsr()
def hops(M, seeds, k=40):
    d=np.full(N,99); d[seeds]=0; f=np.zeros(N,np.float32); f[seeds]=1
    for h in range(1,k+1):
        nf=(M@f>0)&(d==99); d[nf]=h; f=nf.astype(np.float32)
        if not nf.any(): break
    return d
down=hops(A,inp); up=hops(AT,KC)
np.save('logs/hops.npy',np.stack([down,up]))
for k in [2,3,4,5,6,8,99]:
    s=(down<=k)&(up<=k) if k<99 else (down<99)&(up<99)
    sub=W[s][:,s]
    print(k, s.sum(), sub.nnz)
