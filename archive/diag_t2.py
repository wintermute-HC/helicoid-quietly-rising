import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os; os.environ["TEACHER"] = "2"; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from tower_env import make_tower_env
from twist_env import wrap90
from learn_common import eef, gap
from test_markov import MarkovTower, REL
env = make_tower_env(cams=("agentview",), size=32); np.random.seed(0); o = env.reset()
ex = MarkovTower(env, REL); n = 0
while not ex.done and n < 3000 and ex.stuck < 1:
    o, *_ = env.step(ex.act(o)); n += 1
    if n % 100 == 0 and ex.stage == "act" and ex.tr is not None:
        tr = ex.tr; e, yE = eef(o); held = tr.off is not None
        if not held and not tr.held_once: ph, dxy, dz, dy = "掴む前", np.linalg.norm(tr.S[:2]-e[:2]), e[2]-(tr.S[2]+0.005), wrap90(tr.yS-yE)
        elif held: ph, dxy, dz, dy = ("運ぶ" if tr.lifted else "持上"), np.linalg.norm(ex.D[:2]-tr.S[:2]), tr.S[2]-(ex.D[2]+0.003), wrap90(ex.ygoal-tr.yS)
        else: ph, dxy, dz, dy = "離後", 0.0, 0.0, 0.0
        print(f"step {n} 石{ex.k+1} {ph} na={ex.na} 横ずれ={dxy*1000:.1f}mm 高さの差={dz*1000:.1f}mm 角度={np.degrees(dy):+.1f}° 指={gap(o):.3f}", flush=True)
    elif n % 100 == 0: print(f"step {n} 石{ex.k+1} {ex.stage}", flush=True)
print("終了: step", n, "stuck", ex.stuck, "done", ex.done, flush=True)
