import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json; os.environ["MUJOCO_GL"]="egl"; os.environ["PYOPENGL_PLATFORM"]="egl"
import numpy as np
from tower_env import make_tower_env
from tower_expert import TowerExpert, tower_report
from verify import check
u = np.array([1, 1])/np.sqrt(2); s = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
COMP = [dict(color=c, rel_deg=r, dx_mm=float(o[0]), dy_mm=float(o[1])) for c, r, o in zip("BWBWB", [0,18,18,18,18], [(0,0)] + [tuple(s*u)]*4)]
v = check(COMP); print("事前検査:", "合格" if v["ok"] else "不合格", "余裕", v["analysis"]["min_margin_mm"], "mm", flush=True)
env = make_tower_env(cams=("agentview", "artview"), size=320); np.random.seed(700); o = env.reset(); sim = env.sim
ring = {sim.model.geom_name2id(f"ring_seg{i}") for i in range(48)}
ex = TowerExpert(env, COMP); hits = {}; n = 0; frames = []
while not ex.done and n < 3000:
    o, *_ = env.step(ex.act(o)); n += 1
    for c in sim.data.contact[:sim.data.ncon]:
        if (c.geom1 in ring) ^ (c.geom2 in ring):
            oth = c.geom2 if c.geom1 in ring else c.geom1; nm = sim.model.geom_id2name(oth) or str(oth)
            if "table" not in nm: hits[nm] = hits.get(nm, 0) + 1
    if n % 8 == 0: frames.append(np.concatenate([o["artview_image"][::-1], o["agentview_image"][::-1]], 1))
for _ in range(200): o, *_ = env.step(np.r_[np.zeros(6), -1.0])     # 完成後10秒放置して倒れないか
frames.append(np.concatenate([o["artview_image"][::-1], o["agentview_image"][::-1]], 1)); np.save(f"lean{int(s)}_frames.npy", np.array(frames))
ok, rows = tower_report(env, COMP)
print(json.dumps(dict(steps=n, ok=ok, timeouts=ex.timeouts, ring_contacts=hits, levels=[{k: round(v, 1) for k, v in r.items()} for r in rows]), ensure_ascii=False))
