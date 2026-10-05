import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import glob, json, numpy as np
from learn_common import NumpyMLP
M = "/content/mk"
print("== r3 各回(seed, 成功, 歩数, 詰まり, 段数)")
for f in sorted(glob.glob(f"{M}/r3_w*.log")):
    for l in open(f):
        if l.startswith("{"):
            d = json.loads(l); print(d["seed"], d["ok"], d["steps"], d["stuck"], len(d["levels"]))
E = [np.load(f) for f in sorted(glob.glob(f"{M}/dagger_r3/ep*.npz"))]
X = np.concatenate([e["X"] for e in E]); A = np.concatenate([e["A"] for e in E])
pol = NumpyMLP(f"{M}/pol_r2.npz"); P = np.array([pol(x) for x in X])
ho, li, pg = X[:, 16] > 0.5, X[:, 17] > 0.5, X[:, 4] > 0
cat = {"掴む前": ~ho, "掴んで持ち上げ中": ho & ~li & pg, "運んで置く途中": ho & li & pg, "離した後": ho & (X[:, 4] < 0)}
print(f"\n== 局面別(全 {len(X)} 歩)")
for k, m in cat.items():
    if m.sum() == 0: print(k, 0); continue
    err = np.abs(P[m, :6] - A[m, :6]).mean(0); gdis = (np.sign(P[m, 6]) != np.sign(A[m, 6])).mean()
    print(f"{k}: {m.sum()}歩 ({m.mean():.0%}) | ずれ xyz={np.round(err[:3],3)} 回転={np.round(err[3:6],3)} | 開閉の不一致 {gdis:.1%}")
    print(f"   先生の平均 xyz={np.round(A[m,:3].mean(0),3)} 開閉={A[m,6].mean():+.2f} / 生徒の平均 xyz={np.round(P[m,:3].mean(0),3)} 開閉={P[m,6].mean():+.2f}")
