import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json, time; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from learn_common import Tracker, features, act_done, NumpyMLP
from markov_teacher import teacher
from test_markov import MarkovTower, REL, MAX_ACT
from seeing_expert import SeeingTowerExpert
class HybridTower(MarkovTower):          # 動きは生徒、指を開く瞬間だけ先生の規則
    def act(self, o):
        if self.stage != "act" or self.done: return SeeingTowerExpert.act(self, o)
        if self.tr is None: self.tr = Tracker(self.P0, self.yP0); self.na = 0
        self.tr.update(o)
        if act_done(o, self.tr, self.D) or self.na >= MAX_ACT:
            if self.na >= MAX_ACT: self.stuck += 1
            self.k += 1; self.stage = "view"; self.t = 0; self.tr = None; return SeeingTowerExpert.act(self, o)
        f = features(o, self.tr, self.axis, self.D, self.ygoal); at = teacher(o, self.tr, self.D, self.ygoal)
        a = self.pol(f).copy(); a[6] = at[6]; self.na += 1; self.n_pol = getattr(self, "n_pol", 0) + 1
        self.tr.prev_g = a[6]; return a
pol = NumpyMLP(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for s in map(int, sys.argv[3].split(",")):
    t0 = time.time(); env = make_tower_env(cams=("agentview",), size=32); np.random.seed(s); o = env.reset()
    ex = HybridTower(env, REL, pol=pol, beta=0.0, seed=s); n = 0
    while not ex.done and n < 6000 and ex.stuck < 3: o, *_ = env.step(ex.act(o)); n += 1
    for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    ok, rows = tower_report(env, REL)
    print(json.dumps(dict(seed=s, ok=bool(ok), steps=n, stuck=ex.stuck, refix=sum(ex.fixes.values()), sec=round(time.time()-t0),
          levels=[{k: round(v, 1) for k, v in r.items()} for r in rows])), flush=True)
