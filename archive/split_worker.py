# 分担: 遠くは学習した生徒、石や置き場所から横2cm以内(最後の寄せ・掴む・降ろす・離す)は先生
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json, time; os.environ.pop("TEACHER", None); os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from learn_common import Tracker, features, act_done, NumpyMLP, eef
from markov_teacher import teacher
from test_markov import MarkovTower, REL, MAX_ACT
from seeing_expert import SeeingTowerExpert
NEAR = 0.02
class SplitTower(MarkovTower):
    def act(self, o):
        if self.stage != "act" or self.done: return SeeingTowerExpert.act(self, o)
        if self.tr is None: self.tr = Tracker(self.P0, self.yP0); self.na = 0
        self.tr.update(o)
        if act_done(o, self.tr, self.D) or self.na >= MAX_ACT:
            if self.na >= MAX_ACT: self.stuck += 1
            self.k += 1; self.stage = "view"; self.t = 0; self.tr = None; return SeeingTowerExpert.act(self, o)
        f = features(o, self.tr, self.axis, self.D, self.ygoal); at = teacher(o, self.tr, self.D, self.ygoal)
        tr = self.tr; e, _ = eef(o); held = tr.off is not None
        if not held and not tr.held_once: near = np.linalg.norm(tr.S[:2] - e[:2]) < NEAR or tr.prev_g > 0
        elif held and tr.lifted: near = np.linalg.norm(self.D[:2] - tr.S[:2]) < NEAR
        else: near = False
        a = at if near else self.pol(f).copy()
        self.na += 1; self.n_all = getattr(self, "n_all", 0) + 1; self.n_pol = getattr(self, "n_pol", 0) + (0 if near else 1)
        self.tr.prev_g = a[6]; return a
pol = NumpyMLP(sys.argv[1])
for s in map(int, sys.argv[2].split(",")):
    t0 = time.time(); env = make_tower_env(cams=("agentview",), size=32); np.random.seed(s); o = env.reset()
    ex = SplitTower(env, REL, pol=pol, beta=0.0, seed=s); n = 0
    while not ex.done and n < 6000 and ex.stuck < 3: o, *_ = env.step(ex.act(o)); n += 1
    for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    ok, rows = tower_report(env, REL)
    print(json.dumps(dict(seed=s, ok=bool(ok), steps=n, stuck=ex.stuck, student_share=round(getattr(ex, "n_pol", 0)/max(1, getattr(ex, "n_all", 1)), 3),
          sec=round(time.time()-t0), levels=[{k: round(v, 1) for k, v in r.items()} for r in rows])), flush=True)
