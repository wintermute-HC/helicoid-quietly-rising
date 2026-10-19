import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from seeing_expert import SeeingTowerExpert
from learn_common import Tracker, features, act_done
if os.environ.get('TEACHER') == '2': from markov_teacher2 import teacher
else: from markov_teacher import teacher
REL = [0, 22.5, 22.5, 22.5, 22.5]; MAX_ACT = 600
class MarkovTower(SeeingTowerExpert):
    def __init__(self, env, comp, pol=None, beta=1.0, noise=0.0, seed=0):
        super().__init__(env, comp); self.pol, self.beta, self.noise = pol, beta, noise
        self.rng = np.random.default_rng(seed); self.tr = None; self.na = 0; self.stuck = 0; self.X, self.A = [], []
    def act(self, o):
        if self.stage != "act" or self.done: return super().act(o)
        if self.tr is None: self.tr = Tracker(self.P0, self.yP0); self.na = 0
        self.tr.update(o)
        if act_done(o, self.tr, self.D) or self.na >= MAX_ACT:
            if self.na >= MAX_ACT: self.stuck += 1
            self.k += 1; self.stage = "view"; self.t = 0; self.tr = None; return super().act(o)
        f = features(o, self.tr, self.axis, self.D, self.ygoal); at = teacher(o, self.tr, self.D, self.ygoal)
        self.X.append(f); self.A.append(at.copy()); self.na += 1
        a = at if (self.pol is None or self.rng.random() < self.beta) else self.pol(f)
        if self.noise > 0: a = a.copy(); a[:6] = np.clip(a[:6] + min(1.0, float(np.abs(at[:3]).max()))*self.rng.normal(0, self.noise, 6), -1, 1)
        self.tr.prev_g = a[6]; return a
def run(seed):
    env = make_tower_env(cams=("agentview",), size=32); np.random.seed(seed); o = env.reset()
    ex = MarkovTower(env, REL); n = 0
    while not ex.done and n < 6000 and ex.stuck < 3: o, *_ = env.step(ex.act(o)); n += 1
    for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    ok, rows = tower_report(env, REL)
    return dict(seed=seed, ok=bool(ok), steps=n, stuck=ex.stuck, refix=sum(ex.fixes.values()), samples=len(ex.X),
                levels=[{k: round(v, 1) for k, v in r.items()} for r in rows])
if __name__ == "__main__":
    for s in map(int, sys.argv[1:]): print(json.dumps(run(s)), flush=True)
