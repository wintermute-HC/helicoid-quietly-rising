import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]

import os, sys, json, time, shutil, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
_CWD = os.getcwd(); CODE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, CODE); os.chdir(CODE)
from test_markov import MarkovTower, REL, MAX_ACT
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from learn_common import Tracker, features, act_done, NumpyMLP, eef
from markov_teacher import teacher
from expert import R_DOWN, Rz, mat2aa
from twist_env import quat2mat
DRV = os.path.join(_CWD, os.environ.get("HELICOID_DRIVE", "drive_out"), "bc_v5")
POL = os.environ.get("POL", f"{DRV}/pol_r2.npz")
SEED = 30001
nrm = lambda v: float(np.linalg.norm(v))
def tilt_deg(o):
    e, yE = eef(o); R = quat2mat(o["robot0_eef_quat"])
    return float(np.degrees(nrm(mat2aa(Rz(yE) @ R_DOWN @ R.T))))
class Diag(MarkovTower):
    def __init__(self, *a, **k): super().__init__(*a, **k); self.log = []
    def act(self, o):
        if self.stage != "act" or self.done: return super().act(o)
        if self.tr is None: self.tr = Tracker(self.P0, self.yP0); self.na = 0
        self.tr.update(o)
        if act_done(o, self.tr, self.D) or self.na >= MAX_ACT:
            if self.na >= MAX_ACT: self.stuck += 1; self.log.append(dict(ev="STUCK", k=int(self.k)))
            self.k += 1; self.stage = "view"; self.t = 0; self.tr = None; return super().act(o)
        f = features(o, self.tr, self.axis, self.D, self.ygoal); at = teacher(o, self.tr, self.D, self.ygoal)
        a = at if self.pol is None else self.pol(f)
        e, _ = eef(o); S = self.tr.S; D = self.D
        self.log.append(dict(k=int(self.k), na=int(self.na), held=bool(self.tr.off is not None),
            ho=bool(self.tr.held_once), lift=bool(self.tr.lifted),
            d_grasp=nrm(S + np.array([0, 0, 0.005]) - e)*1000,
            d_place=nrm(np.r_[D[:2] - S[:2], D[2] + 0.003 - S[2]])*1000,
            tilt=tilt_deg(o), at_p=nrm(at[:3]), a_p=nrm(a[:3]),
            dpos=nrm(a[:3] - at[:3]), drot=nrm(a[3:6] - at[3:6]), gt=float(at[6]), ga=float(a[6])))
        self.na += 1; self.tr.prev_g = a[6]; return a
for mode in ("student", "teacher"):
    t0 = time.time()
    env = make_tower_env(cams=("agentview",), size=32); np.random.seed(SEED); o = env.reset()
    ex = Diag(env, REL, pol=(NumpyMLP(POL) if mode == "student" else None), beta=0.0); n = 0
    while not ex.done and n < 6000 and ex.stuck < 3:
        o, *_ = env.step(ex.act(o)); n += 1
        if n % 300 == 0: print(f"{mode} step {n} stuck {ex.stuck} k {ex.k} {time.time()-t0:.0f}s", flush=True)
    for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    ok, rows = tower_report(env, REL)
    out = f"/content/diag_1004_{mode}.json"
    json.dump(dict(mode=mode, seed=SEED, ok=bool(ok), steps=n, stuck=int(ex.stuck), log=ex.log), open(out, "w"))
    shutil.copy(out, DRV)
    print(f"{mode} 完了: ok={ok} steps={n} stuck={ex.stuck} {time.time()-t0:.0f}s", flush=True)
print("DIAG_DONE", flush=True)
