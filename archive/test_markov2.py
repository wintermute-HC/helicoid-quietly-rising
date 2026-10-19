import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json, time; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from test_markov import MarkovTower, REL
def run(seed):
    t0 = time.time(); env = make_tower_env(cams=("agentview",), size=32); np.random.seed(seed); o = env.reset()
    ex = MarkovTower(env, REL); n = 0
    while not ex.done and n < 6000 and ex.stuck < 3:
        o, *_ = env.step(ex.act(o)); n += 1
        if n % 500 == 0:
            print(json.dumps(dict(seed=seed, step=n, sec=round(time.time()-t0), level=ex.k+1, stage=ex.stage, na=ex.na, stuck=ex.stuck,
                  held=(ex.tr is not None and ex.tr.off is not None), eef_z=round(float(o["robot0_eef_pos"][2]), 3))), flush=True)
    for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    ok, rows = tower_report(env, REL)
    print(json.dumps(dict(seed=seed, RESULT=bool(ok), steps=n, sec=round(time.time()-t0), stuck=ex.stuck, refix=sum(ex.fixes.values()),
          levels=[{k: round(v, 1) for k, v in r.items()} for r in rows])), flush=True)
if __name__ == "__main__":
    for s in map(int, sys.argv[1:]): run(s)
    print("ALL_DONE", flush=True)
