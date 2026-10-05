import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json; os.environ["MUJOCO_GL"]="egl"; os.environ["PYOPENGL_PLATFORM"]="egl"
import numpy as np
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from seeing_expert import SeeingTowerExpert
REL = [0, 22.5, 22.5, 22.5, 22.5]; seed = int(sys.argv[1]) if len(sys.argv) > 1 else 700
env = make_tower_env(cams=("agentview", "artview"), size=320); np.random.seed(seed); o = env.reset()
ex = SeeingTowerExpert(env, REL); n = 0; frames = []
while not ex.done and n < 4000:
    o, *_ = env.step(ex.act(o)); n += 1
    if n % 10 == 0: frames.append(np.concatenate([o["artview_image"][::-1], o["agentview_image"][::-1]], 1))
for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
frames.append(np.concatenate([o["artview_image"][::-1], o["agentview_image"][::-1]], 1)); np.save("seeing_frames.npy", np.array(frames))
ok, rows = tower_report(env, REL)       # 正解は採点にだけ使う
print(json.dumps(dict(seed=seed, steps=n, ok=ok, timeouts=ex.timeouts, perception_log=ex.log,
                      levels=[{k: round(v, 1) for k, v in r.items()} for r in rows]), ensure_ascii=False))
