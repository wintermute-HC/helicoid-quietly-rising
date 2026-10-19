import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json, time; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from learn_common import NumpyMLP, F_DIM
from test_markov import MarkovTower, REL
pol_path, beta, sigma, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
seeds = [int(s) for s in sys.argv[5].split(",")]
pol = None if pol_path == "none" else NumpyMLP(pol_path); os.makedirs(out, exist_ok=True)
for s in seeds:
    fn = f"{out}/ep{s:05d}.npz"
    if os.path.exists(fn): continue
    t0 = time.time(); env = make_tower_env(cams=("agentview",), size=32); np.random.seed(s); o = env.reset()
    ex = MarkovTower(env, REL, pol=pol, beta=beta, noise=(sigma if s % 2 else 0.0), seed=s); n = 0
    while not ex.done and n < 6000 and ex.stuck < 3: o, *_ = env.step(ex.act(o)); n += 1
    for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    ok, rows = tower_report(env, REL)
    np.savez_compressed(fn, X=np.array(ex.X, np.float32).reshape(-1, F_DIM), A=np.array(ex.A, np.float32).reshape(-1, 7), ok=ok, steps=n, stuck=ex.stuck)
    print(json.dumps(dict(seed=s, ok=bool(ok), steps=n, stuck=ex.stuck, samples=len(ex.X), sec=round(time.time() - t0),
          levels=[{k: round(v, 1) for k, v in r.items()} for r in rows])), flush=True)
    try: env.close()
    except Exception: pass
