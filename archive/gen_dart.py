# DART方式データ生成: 実行する行動にノイズを加えて軌道を乱し、ラベルには「乱れた状態から見たエキスパートの修正行動」を記録する
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json; os.environ["MUJOCO_GL"]="egl"; os.environ["PYOPENGL_PLATFORM"]="egl"; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from twist_env import make_env
from expert import Expert, evaluate
wid, nw, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
N_EP = int(sys.argv[4]) if len(sys.argv) > 4 else 400
os.makedirs(OUT, exist_ok=True)
NOISY_PH = {0, 3, 4}                   # 接近・持ち上げ・運搬の間だけ乱す(把持・設置の精密動作中は乱さない)
SIG_POS, SIG_ROT = 0.25, 0.15          # 行動ノイズ(正規化単位)。1.0 = 1stepあたり最大5cm
def job(i):                             # エピソードi: 角度は0〜89°の整数を一様に、半数はノイズなし
    rng = np.random.RandomState(900000 + i)
    return i, int(rng.randint(0, 90)), bool(i % 2)
env = make_env(cams=("agentview", "robot0_eye_in_hand")); log = open(f"{OUT}/worker{wid}.log", "a")
KEYS = ["agentview","wrist","eef_pos","eef_quat","grip_qpos","action","cubeA","cubeB"]
for i, a, noisy in [job(i) for i in range(N_EP)][wid::nw]:
    fn = f"{OUT}/ep{i:04d}_a{a:02d}_{'dart' if noisy else 'clean'}.npz"
    if os.path.exists(fn): continue
    seed, tries = 3_000_000 + i*10, 0
    while True:
        np.random.seed(seed); rng = np.random.RandomState(seed + 1)
        o = env.reset(); ex = Expert(a); rec = {x: [] for x in KEYS}; n = 0
        while not ex.done and n < 500:
            ph = ex.ph; act = ex.act(o)                       # ラベル = 現在(乱れた)状態からの修正行動
            for x, v in zip(KEYS, [o["agentview_image"], o["robot0_eye_in_hand_image"], o["robot0_eef_pos"], o["robot0_eef_quat"],
                                   o["robot0_gripper_qpos"], act, np.r_[o["cubeA_pos"],o["cubeA_quat"]], np.r_[o["cubeB_pos"],o["cubeB_quat"]]]):
                rec[x].append(v)
            ex_act = act.copy()
            if noisy and ph in NOISY_PH:
                k = min(1.0, float(np.abs(act[:3]).max()))    # 目標に近づくほど乱れを弱める(収束を妨げない)
                ex_act[:3] = np.clip(ex_act[:3] + k*rng.normal(0, SIG_POS, 3), -1, 1)
                ex_act[3:6] = np.clip(ex_act[3:6] + k*rng.normal(0, SIG_ROT, 3), -1, 1)
            o, *_ = env.step(ex_act); n += 1                  # 実行は乱した行動
        for _ in range(20): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
        m = evaluate(o, a)
        if m["success"] or tries >= 3: break
        tries += 1; seed += 7919
    if m["success"]:
        np.savez_compressed(fn, **{x: np.asarray(v) for x, v in rec.items()}, angle=a, seed=seed, dart=noisy,
                            instruction=f"stack the red cube on the green cube, rotated {a:g} degrees")
    log.write(json.dumps(dict(i=i, angle=a, dart=noisy, seed=seed, steps=n, tries=tries, timeouts=ex.timeouts,
                              **{k: (v if isinstance(v, bool) else round(float(v), 2)) for k, v in m.items()})) + "\n"); log.flush()
log.write("WORKER_DONE\n"); env.close()
