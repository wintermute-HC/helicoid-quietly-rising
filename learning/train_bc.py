# 第1段の学習: 教材(円環座標の特徴量 → 先生の行動)から小型の方策(全結合3層)を学ぶ。成功したエピソードだけを使う。
import sys, glob, json, numpy as np, torch, torch.nn as nn
d, out = sys.argv[1], sys.argv[2]; epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
eps, good = [], []
for part in d.split(","):                                   # 例: /content/bc_data2,/content/dagger1*3 (*3 = 3倍の重み)
    p, w = (part.split("*") + ["1"])[:2]
    E = [np.load(f) for f in sorted(glob.glob(f"{p}/ep*.npz"))]; eps += E
    good += [e for e in E if len(e["X"]) and (bool(e["ok"]) or "dagger" in p)] * int(w)   # 介入の教材は先生の行動なので失敗回も使う
print(f"episodes {len(eps)} / used(ok) {len(good)}", flush=True)
nv = max(1, len(good)//10); tr_, va_ = (good[nv:], good[:nv]) if len(good) >= 10 else (good, good[:1])
cat = lambda L, k: np.concatenate([e[k] for e in L])
Xt, At, Xv, Av = cat(tr_, "X"), cat(tr_, "A"), cat(va_, "X"), cat(va_, "A")
mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
dev = "cuda" if torch.cuda.is_available() else "cpu"
T = lambda x: torch.tensor(x, device=dev)
Xt_, At_, Xv_, Av_ = T((Xt-mu)/sd), T(At), T((Xv-mu)/sd), T(Av)
net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 7)).to(dev)
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
def loss_fn(p, a):
    return ((p[:, :6] - a[:, :6])**2).mean() + nn.functional.binary_cross_entropy_with_logits(p[:, 6], (a[:, 6] > 0).float())
for ep in range(epochs):
    net.train(); perm = torch.randperm(len(Xt_), device=dev)
    for i in range(0, len(perm), 1024):
        j = perm[i:i+1024]; l = loss_fn(net(Xt_[j]), At_[j]); opt.zero_grad(); l.backward(); opt.step()
    sch.step(); net.eval()
    with torch.no_grad():
        pv = net(Xv_); lv = loss_fn(pv, Av_).item(); gacc = ((pv[:, 6] > 0) == (Av_[:, 6] > 0)).float().mean().item()
    if ep % 5 == 0 or ep == epochs-1: print(json.dumps(dict(epoch=ep, train=round(l.item(), 5), val=round(lv, 5), grip_acc=round(gacc, 4))), flush=True)
L = [m for m in net if isinstance(m, nn.Linear)]
np.savez(out, n=len(L), mu=mu, sd=sd, **{f"W{i}": m.weight.detach().cpu().numpy().T for i, m in enumerate(L)}, **{f"b{i}": m.bias.detach().cpu().numpy() for i, m in enumerate(L)})
print("saved", out, "samples", len(Xt), flush=True)
