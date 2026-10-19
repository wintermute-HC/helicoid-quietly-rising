# 「静昇」撮影: 作品カメラ(artview, 1280x720)で見て積む全工程を実時間で記録 → 完成後に腕を退け、塔の周りを一周する
# 出力: film/main.mp4(本編), film/orbit.mp4(周回), film/look_k*.png(各石の知覚画像), film/report.json
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json, subprocess; os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np, robosuite as suite
from PIL import Image, ImageDraw
from tower_env import RingTower, RING_C, TABLE_Z, lookat_quat, H
from tower_expert import tower_report_by_height
from seeing_expert import SeeingTowerExpert, VIEW_POS
from expert import R_DOWN, mat2aa
from twist_env import quat2mat
import perception
REL = [float(x) for x in os.environ.get("REL", "0,22.5,22.5,22.5,22.5").split(",")]
SEED = int(os.environ.get("SEED", 700)); W, HH = 1280, 720; OUT = os.environ.get("OUT", "film"); FPS = 20
os.makedirs(OUT, exist_ok=True)
env = RingTower(robots="Panda", controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
                has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, control_freq=20, horizon=8000, ignore_done=True)
np.random.seed(SEED); o = env.reset()
cam = env.sim.model.camera_name2id("artview"); m = env.sim.model
shot = lambda: env.sim.render(width=W, height=HH, camera_name="artview")[::-1]
def writer(path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{HH}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", path], stdin=subprocess.PIPE)
ACTOR = os.environ.get('ACTOR', 'seeing')
if ACTOR == 'seeing': ex = SeeingTowerExpert(env, REL)
elif ACTOR == 'split':
    from split_tower import SplitTower
    from learn_common import NumpyMLP
    ex = SplitTower(env, REL, pol=NumpyMLP(os.environ['POL']), beta=0.0, seed=SEED)
else:
    from test_markov import MarkovTower
    from learn_common import NumpyMLP
    pol = NumpyMLP(os.environ['POL']) if ACTOR == 'student' else None
    ex = MarkovTower(env, REL, pol=pol, beta=(0.0 if pol else 1.0), seed=SEED)
print('ACTOR', ACTOR, os.environ.get('POL', ''), flush=True)
orig_look = ex._look; looks = []
def look_and_record():                               # 知覚の瞬間: 真上カメラ画像に検出結果を描き込んで保存
    rgb, depth = perception.capture(env); obs = perception.observe(env); orig_look()
    im = Image.fromarray(rgb.copy()); d = ImageDraw.Draw(im); f = (perception.RES/2)/np.tan(np.radians(perception.TOPCAM_FOVY)/2); c = (perception.RES-1)/2
    for ob in obs:
        dist = perception.TOPCAM_Z - ob["z_top"]; u = c + (ob["xy"][0]-RING_C[0])*f/dist; v = c - (ob["xy"][1]-RING_C[1])*f/dist
        half = H*f/dist; ca, sa = np.cos(ob["yaw"]), np.sin(ob["yaw"])
        pts = [(u + half*(ca*px - sa*py), v - half*(sa*px + ca*py)) for px, py in ((-1,-1),(1,-1),(1,1),(-1,1))]
        col = (255, 60, 40) if ob["in_ring"] else (40, 160, 255)
        d.polygon(pts, outline=col, width=3); d.text((u+half+4, v-8), f'{ob["color"]} L{ob["level"]} {np.degrees(ob["yaw"]):+.1f}°', fill=col)
    im.save(f"{OUT}/look_k{ex.k}.png"); looks.append(dict(k=ex.k, detections=[dict(xy=[round(float(v), 4) for v in ob["xy"]], color=ob["color"], level=ob["level"],
                                                          yaw_deg=round(float(np.degrees(ob["yaw"])), 2), in_ring=ob["in_ring"]) for ob in obs]))
ex._look = look_and_record
# ---- 本編 ----
p = writer(f"{OUT}/main.mp4"); n = 0
for _ in range(FPS*2): p.stdin.write(shot().tobytes())                       # 冒頭2秒の静止(初期状態)
while not ex.done and n < 6000 and getattr(ex, 'stuck', 0) < 3:
    o, *_ = env.step(ex.act(o)); n += 1; p.stdin.write(shot().tobytes())
    if n % 200 == 0: print("step", n, flush=True)
for _ in range(40): o, *_ = env.step(np.r_[np.zeros(6), -1.0]); p.stdin.write(shot().tobytes())
PARK = VIEW_POS + [0, 0, 0.15]                                                # 腕を塔から退ける
for _ in range(120):
    R = quat2mat(o["robot0_eef_quat"]); a = np.zeros(7); a[:3] = np.clip((PARK - o["robot0_eef_pos"])*10, -1, 1)
    a[3:6] = np.clip(mat2aa(R_DOWN @ R.T)*2, -1, 1); a[6] = -1; o, *_ = env.step(a); p.stdin.write(shot().tobytes())
for _ in range(FPS*3): p.stdin.write(shot().tobytes())                         # 完成3秒静止
p.stdin.close(); p.wait()
ok, rows = tower_report_by_height(env, REL)
Image.fromarray(shot()).save(f"{OUT}/final_art.png")
# ---- 周回(ロボット側を避けて手前を通る240°の弧, 14秒, 始終はゆっくり) ----
np.save(f"{OUT}/final_state.npy", env.sim.get_state().flatten())
tgt = np.array([RING_C[0], RING_C[1], TABLE_Z + 0.11]); rad, z = 0.60, TABLE_Z + 0.30
p = writer(f"{OUT}/orbit.mp4"); N = FPS*14
for i in range(N):
    s_ = (1 - np.cos(np.pi*i/(N-1)))/2; th = np.radians(-120 + 240*s_)
    pos = np.r_[tgt[:2] + rad*np.array([np.cos(th), np.sin(th)]), z]
    m.cam_pos[cam] = pos; m.cam_quat[cam] = lookat_quat(pos, tgt); env.sim.forward(); p.stdin.write(shot().tobytes())
p.stdin.close(); p.wait()
rep = dict(student_share=round(getattr(ex, 'n_pol', 0)/max(1, getattr(ex, 'n_all', 1)), 3), actor=ACTOR, pol=os.environ.get('POL', ''), seed=SEED, rel_deg=REL, steps=n, ok=ok, timeouts=ex.timeouts, levels=[{k: round(v, 2) for k, v in r.items()} for r in rows], looks=looks)
json.dump(rep, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(dict(ok=ok, steps=n, levels=rep["levels"]), ensure_ascii=False), flush=True)
os._exit(0)
