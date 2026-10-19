# 第1段の追加学習(DAgger・介入型): 学習した方策に積ませ、詰まった所で先生(見て積む制御)が引き継いで立て直す。
# 引き継いだ区間の「方策が実際に迷い込んだ状態 → 先生の行動」を新しい教材として記録する(先生が見たことのない状態を補う)。
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from tower_env import make_tower_env, TABLE_Z
from tower_expert import tower_report_by_height as tower_report
from seeing_expert import SeeingTowerExpert
from learn_common import Tracker, features, act_done, NumpyMLP, eef
REL = [0, 22.5, 22.5, 22.5, 22.5]
class DaggerTower(SeeingTowerExpert):
    def __init__(self, env, comp, pol):
        super().__init__(env, comp); self.pol, self.tr, self.na, self.expert, self.hist = pol, None, 0, False, []
        self.X, self.A, self.interventions = [], [], 0
    def _takeover(self, o):                                       # 方策の記憶から、先生の局面を決めて引き継ぐ
        e, yE = eef(o); tr = self.tr; safe = max(TABLE_Z + 0.20, self.D[2] + 0.133 - 0.003 + 0.003)
        if tr.off is not None:
            self.grip_off, self.dyaw = tr.off.copy(), tr.dy
            self.ph = 3 if not tr.lifted else (4 if e[2] > safe - 0.03 else 5)
        else: self.ph = 7 if tr.held_once else 0
        self.t = 0; self.yaw_t = None; self.hold = None; self.expert = True; self.interventions += 1
    def act(self, o):
        if self.stage != "act" or self.done:
            self.tr = None; self.expert = False; self.hist = []; return super().act(o)
        if self.tr is None: self.tr = Tracker(self.P0, self.yP0); self.na = 0; self.hist = []
        self.tr.update(o); f = features(o, self.tr, self.axis, self.D, self.ygoal); e, _ = eef(o)
        if not self.expert:
            if act_done(o, self.tr, self.D):
                self.k += 1; self.stage = "view"; self.t = 0; self.tr = None; return super().act(o)
            self.hist.append(e.copy()); self.na += 1
            stalled = len(self.hist) > 40 and np.linalg.norm(self.hist[-1] - self.hist[-31]) < 0.002
            if stalled or self.na >= 300: self._takeover(o)
            else:
                a = self.pol(f); self.tr.prev_g = a[6]; return a
        a = super().act(o)                                          # 先生の行動(局面が終われば先生が見る局面へ移す)
        self.X.append(f); self.A.append(a.copy()); self.tr.prev_g = a[6]
        return a
def run(seed, pol, out):
    env = make_tower_env(cams=("agentview",), size=32); np.random.seed(seed); o = env.reset()
    ex = DaggerTower(env, REL, pol); n = 0
    while not ex.done and n < 6000: o, *_ = env.step(ex.act(o)); n += 1
    for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    ok, rows = tower_report(env, REL)
    np.savez_compressed(out, X=np.array(ex.X, np.float32).reshape(-1, 18), A=np.array(ex.A, np.float32).reshape(-1, 7), ok=ok, steps=n)
    return dict(seed=seed, ok=bool(ok), steps=n, interventions=ex.interventions, samples=len(ex.X))
if __name__ == "__main__":
    pol = NumpyMLP(sys.argv[1]); s0, k, d = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; os.makedirs(d, exist_ok=True)
    for s in range(s0, s0 + k):
        fn = f"{d}/ep{s:05d}.npz"
        if not os.path.exists(fn): print(json.dumps(run(s, pol, fn)), flush=True)
