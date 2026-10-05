import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json; os.environ["MUJOCO_GL"]="egl"; os.environ["PYOPENGL_PLATFORM"]="egl"
import numpy as np
from ring_env import make_ring_env, RING_C
from ring_expert import RingExpert
from expert import evaluate
REL = float(sys.argv[1]) if len(sys.argv) > 1 else 22.5
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
env = make_ring_env()
res = []
for ep in range(N):
    np.random.seed(500 + ep); o = env.reset(); sim = env.sim
    ring_ids = {sim.model.geom_name2id(f"ring_seg{i}") for i in range(32)}
    table_like = {i for i in range(sim.model.ngeom) if "table" in (sim.model.geom_id2name(i) or "")}
    ex = RingExpert([("cubeB", RING_C, 0.0), ("cubeA", "cubeB", ("rel", "cubeB", np.radians(REL)))])
    frames = []; hits = {}; n = 0
    while not ex.done and n < 900:
        a = ex.act(o); o, *_ = env.step(a); n += 1
        for c in sim.data.contact[:sim.data.ncon]:
            g1, g2 = c.geom1, c.geom2
            if (g1 in ring_ids) ^ (g2 in ring_ids):
                other = g2 if g1 in ring_ids else g1
                if other in table_like: continue
                nm = sim.model.geom_id2name(other) or f"geom{other}"
                hits[nm] = hits.get(nm, 0) + 1
        if ep == 0 and n % 6 == 0: frames.append((o["agentview_image"][::-1], o["topview_image"][::-1]))
    for _ in range(20): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
    m = evaluate(o, REL)
    dB = np.linalg.norm(o["cubeB_pos"][:2] - RING_C) * 1000
    r = dict(ep=ep, steps=n, timeouts=ex.timeouts, ring_contacts=hits, B_center_mm=round(dB,1),
             **{k: (round(float(v),2) if not isinstance(v,bool) else v) for k,v in m.items()})
    res.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    if ep == 0: np.save("frames0.npy", np.array([np.concatenate(f, 1) for f in frames]))
