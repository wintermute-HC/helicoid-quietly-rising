# 第1段の評価: 運動の局面を、学習した方策に任せて五段を積む(上位の知覚・検査・置き直しは見て積む制御のまま)。
# 正解(シミュレータの真値)は採点にだけ使う。
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from seeing_expert import SeeingTowerExpert
from learn_common import Tracker, features, act_done, NumpyMLP
REL = [0, 22.5, 22.5, 22.5, 22.5]; MAX_ACT = 600
class StudentTower(SeeingTowerExpert):
    def __init__(self, env, comp, pol):
        super().__init__(env, comp); self.pol, self.tr, self.na, self.act_steps, self.stuck = pol, None, 0, 0, 0
    def act(self, o):
        if self.stage != "act" or self.done: return super().act(o)
        if self.tr is None: self.tr = Tracker(self.P0, self.yP0); self.na = 0
        self.tr.update(o)
        if act_done(o, self.tr, self.D) or self.na >= MAX_ACT:          # 置いて離れた(または打ち切り): 見る局面へ
            if self.na >= MAX_ACT: self.stuck += 1
            self.k += 1; self.stage = "view"; self.t = 0; self.tr = None; return super().act(o)
        a = self.pol(features(o, self.tr, self.axis, self.D, self.ygoal)); self.tr.prev_g = a[6]; self.na += 1; self.act_steps += 1
        return a
def run(seed, pol, film=None):
    env = make_tower_env(cams=("artview",) if film else ("agentview",), size=320 if film else 32); np.random.seed(seed); o = env.reset()
    ex = StudentTower(env, REL, pol); n, frames = 0, []
    while not ex.done and n < 6000 and ex.stuck < 3:              # 3回詰まったら打ち切り(時間の節約)
        o, *_ = env.step(ex.act(o)); n += 1
        if film and n % 4 == 0: frames.append(o["artview_image"][::-1])
    for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    ok, rows = tower_report(env, REL)
    if film: np.save(film, np.array(frames))
    return dict(seed=seed, ok=bool(ok), steps=n, policy_steps=ex.act_steps, stuck=ex.stuck, refix=sum(ex.fixes.values()),
                levels=[{k: round(v, 1) for k, v in r.items()} for r in rows])
if __name__ == "__main__":
    pol = NumpyMLP(sys.argv[1]); s0, k = int(sys.argv[2]), int(sys.argv[3]); film = sys.argv[4] if len(sys.argv) > 4 else None
    for s in range(s0, s0 + k): print(json.dumps(run(s, pol, film)), flush=True)
