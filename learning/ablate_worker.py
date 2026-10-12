# 切り分け実験(10/12): 方策を生徒単独(先生の介入なし)で評価する。mk_worker.py と同じ条件(奇数の種は乱れ0.05)。
# 入力の一部を外した方策(npz に keep がある)にも対応する。使い方: python ablate_worker.py <方策npz> <出力フォルダ> <種,種,...>
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json, time; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from learn_common import NumpyMLP, F_DIM
from test_markov import MarkovTower, REL
class Masked(NumpyMLP):
    def __init__(self, path):
        super().__init__(path); z = np.load(path); self.keep = z["keep"] if "keep" in z.files else np.arange(F_DIM)
    def __call__(self, f): return super().__call__(np.asarray(f)[self.keep])
pol_path, out = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in sys.argv[3].split(",")]
pol = Masked(pol_path); os.makedirs(out, exist_ok=True)
for s in seeds:
    fn = f"{out}/ep{s:05d}.json"
    if os.path.exists(fn): continue
    t0 = time.time(); env = make_tower_env(cams=("agentview",), size=32); np.random.seed(s); o = env.reset()
    ex = MarkovTower(env, REL, pol=pol, beta=0.0, noise=(0.05 if s % 2 else 0.0), seed=s); n = 0
    while not ex.done and n < 6000 and ex.stuck < 3: o, *_ = env.step(ex.act(o)); n += 1
    for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    ok, rows = tower_report(env, REL)
    r = dict(seed=s, ok=bool(ok), steps=n, stuck=ex.stuck, sec=round(time.time() - t0), levels=[{k: round(v, 2) for k, v in x.items()} for x in rows])
    json.dump(r, open(fn, "w")); print(json.dumps(r), flush=True)
    try: env.close()
    except Exception: pass
