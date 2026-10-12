# 切り分け実験(10/12): v7 と同じ教材・同じ学習法で、入力の一部を外した方策を学習する。乱数の種を指定できる。
# 使い方: python3 ablate_train.py <教材フォルダ,...> <出力npz> <条件 all|notilt|nomm|noextra> <種> [エポック]
import sys, glob, json, numpy as np, torch, torch.nn as nn
d, out, cond, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 40
KEEP = {"all": list(range(29)), "notilt": list(range(18)) + list(range(21, 29)),        # 18〜20=傾き、21〜28=mm拡大の距離
        "nomm": list(range(21)), "noextra": list(range(18))}[cond]
torch.manual_seed(seed); np.random.seed(seed)
eps, good = [], []
for p in d.split(","):
    E = [np.load(f) for f in sorted(glob.glob(f"{p}/ep*.npz"))]; eps += E
    good += [e for e in E if len(e["X"]) and (bool(e["ok"]) or "dagger" in p)]          # train_bc.py と同じ選び方
print(f"episodes {len(eps)} / used {len(good)} / cond {cond} / seed {seed}", flush=True)
nv = max(1, len(good)//10); tr_, va_ = (good[nv:], good[:nv]) if len(good) >= 10 else (good, good[:1])
cat = lambda L, k: np.concatenate([e[k] for e in L])
Xt, At, Xv, Av = cat(tr_, "X")[:, KEEP], cat(tr_, "A"), cat(va_, "X")[:, KEEP], cat(va_, "A")
mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
dev = "cuda" if torch.cuda.is_available() else "cpu"; T = lambda x: torch.tensor(x, device=dev)
Xt_, At_, Xv_, Av_ = T((Xt-mu)/sd), T(At), T((Xv-mu)/sd), T(Av)
net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 7)).to(dev)
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
def loss_fn(p, a): return ((p[:, :6] - a[:, :6])**2).mean() + nn.functional.binary_cross_entropy_with_logits(p[:, 6], (a[:, 6] > 0).float())
g = torch.Generator(device=dev); g.manual_seed(seed)
for ep in range(epochs):
    net.train(); perm = torch.randperm(len(Xt_), device=dev, generator=g)
    for i in range(0, len(perm), 1024):
        j = perm[i:i+1024]; l = loss_fn(net(Xt_[j]), At_[j]); opt.zero_grad(); l.backward(); opt.step()
    sch.step(); net.eval()
    with torch.no_grad(): pv = net(Xv_); lv = loss_fn(pv, Av_).item(); gacc = ((pv[:, 6] > 0) == (Av_[:, 6] > 0)).float().mean().item()
    if ep % 10 == 0 or ep == epochs-1: print(json.dumps(dict(epoch=ep, train=round(l.item(), 5), val=round(lv, 5), grip_acc=round(gacc, 4))), flush=True)
L = [m for m in net if isinstance(m, nn.Linear)]
np.savez(out, n=len(L), mu=mu, sd=sd, keep=np.array(KEEP), **{f"W{i}": m.weight.detach().cpu().numpy().T for i, m in enumerate(L)}, **{f"b{i}": m.bias.detach().cpu().numpy() for i, m in enumerate(L)})
print("saved", out, "samples", len(Xt), "dim", len(KEEP), flush=True)
