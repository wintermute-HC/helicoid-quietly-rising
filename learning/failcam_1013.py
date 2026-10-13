# 失敗の撮影(10/13): 評価と同じ環境・同じ手順で方策を動かし、作品カメラ(artview)で撮る。停止の局面と、目標までの距離、
# 指の指令(生徒と先生)を画面に重ねる。撮影はレンダリングを足すだけで、方策の動き(成否・歩数)は評価と同じになる。
# 使い方: python failcam_1013.py <方策npz> <種> <出力mp4> <題(画面左上の1行目)> [何歩ごとに1コマ(既定4=4倍速)]
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json, subprocess; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tower_env import make_tower_env
from tower_expert import tower_report_by_height as tower_report
from twist_env import wrap90
from learn_common import Tracker, features, act_done, eef, NumpyMLP, F_DIM
from markov_teacher import teacher
from test_markov import MarkovTower, REL, MAX_ACT
pol_path, seed, out, title = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]; SKIP = int(sys.argv[5]) if len(sys.argv) > 5 else 4
W, HH, FPS = 1280, 720, 20
PH = ["接近", "掴む直前", "閉じる途中", "持ち上げ", "運搬・回転", "置く直前", "離した後"]
FONT = next((p for p in ("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                         "/usr/share/fonts/truetype/ipafont/ipag.ttf") if os.path.exists(p)), None)
assert FONT, "日本語の書体がない(Colab では apt-get install -y fonts-ipafont-gothic)"
F1, F2 = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 24)
class Masked(NumpyMLP):
    def __init__(self, path):
        super().__init__(path); z = np.load(path); self.keep = z["keep"] if "keep" in z.files else np.arange(F_DIM)
    def __call__(self, f): return super().__call__(np.asarray(f)[self.keep])
class Filmed(MarkovTower):
    """MarkovTower.act と同じ手順(乱数の消費順も同じ)。画面に重ねる値だけを控える"""
    def __init__(self, *a, **k): super().__init__(*a, **k); self.hud = None
    def act(self, o):
        if self.stage != "act" or self.done: self.hud = None; return super(MarkovTower, self).act(o)
        if self.tr is None: self.tr = Tracker(self.P0, self.yP0); self.na = 0
        self.tr.update(o)
        if act_done(o, self.tr, self.D) or self.na >= MAX_ACT:
            if self.na >= MAX_ACT: self.stuck += 1
            self.k += 1; self.stage = "view"; self.t = 0; self.tr = None; self.hud = None; return super(MarkovTower, self).act(o)
        f = features(o, self.tr, self.axis, self.D, self.ygoal); at = teacher(o, self.tr, self.D, self.ygoal)
        self.X.append(f); self.A.append(at.copy()); self.na += 1
        a = at if (self.pol is None or self.rng.random() < self.beta) else self.pol(f)
        if self.noise > 0: a = a.copy(); a[:6] = np.clip(a[:6] + min(1.0, float(np.abs(at[:3]).max()))*self.rng.normal(0, self.noise, 6), -1, 1)
        tr = self.tr; e, _ = eef(o); held = tr.off is not None
        dg = float(np.linalg.norm(tr.S + np.array([0, 0, 0.005]) - e)); dp = float(np.linalg.norm(np.r_[self.D[:2] - tr.S[:2], self.D[2] + 0.003 - tr.S[2]]))
        if not held and not tr.held_once: ph = 2 if tr.prev_g > 0 else (1 if dg < 0.01 else 0)
        elif held: ph = 3 if not tr.lifted else (4 if (np.linalg.norm(self.D[:2] - tr.S[:2]) > 0.008 or abs(wrap90(self.ygoal - tr.yS)) > 0.02) else 5)
        else: ph = 6
        self.hud = dict(ph=ph, dg=dg, dp=dp, gs=a[6], gt=at[6], na=self.na)
        self.tr.prev_g = a[6]; return a
env = make_tower_env(cams=("agentview",), size=32); np.random.seed(seed); o = env.reset()
ex = Filmed(env, REL, pol=Masked(pol_path), beta=0.0, noise=(0.05 if seed % 2 else 0.0), seed=seed)
enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{HH}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], stdin=subprocess.PIPE)
def frame(n, extra=None):
    im = Image.fromarray(env.sim.render(width=W, height=HH, camera_name="artview")[::-1].copy()); d = ImageDraw.Draw(im)
    h = ex.hud; L = [title, f"{n}歩目　{SKIP}倍速　停止 {ex.stuck}/3"]
    if h:
        grasp_side = h["ph"] <= 2
        L.append(f"局面：{PH[h['ph']]}　この石の運動 {h['na']}/{MAX_ACT}歩")
        L.append(f"掴む位置まで {h['dg']*1000:.1f}mm（先生が閉じる境目 6mm）" if grasp_side else f"置く位置まで {h['dp']*1000:.1f}mm（先生が離す境目 4mm）")
        L.append(f"指の指令　生徒：{'閉じる' if h['gs'] > 0 else '開く'}　先生：{'閉じる' if h['gt'] > 0 else '開く'}")
    else: L.append("知覚(腕を退けて真上から見る)")
    if extra: L.append(extra)
    d.rectangle([16, 16, 760, 30 + 38*len(L)], fill=(245, 243, 238))
    for i, s in enumerate(L):
        col = (190, 40, 30) if ("食い違い" in s or (h and i == 4 and (h["gs"] > 0) != (h["gt"] > 0))) else (28, 27, 26)
        d.text((30, 26 + 38*i), s, font=F1 if i == 0 else F2, fill=col)
    enc.stdin.write(np.asarray(im).tobytes()); return im
n = 0; frame(0); shot = False
while not ex.done and n < 6000 and ex.stuck < 3:
    o, *_ = env.step(ex.act(o)); n += 1
    if n % SKIP == 0 or (not shot and ex.hud and ex.hud["na"] == MAX_ACT - 10):
        im = frame(n)
        if not shot and ex.hud and ex.hud["na"] == MAX_ACT - 10: im.save(out[:-4] + "_stall.png"); shot = True   # 最初の停止の10歩前を静止画に
for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
ok, rows = tower_report(env, REL)
for i in range(FPS*3):
    im = frame(n, f"結果：{'成功' if ok else '失敗'}（{n}歩、停止 {ex.stuck}回）")
    if i == 0: im.save(out[:-4] + "_end.png")
enc.stdin.close(); enc.wait()
print(json.dumps(dict(seed=seed, ok=bool(ok), steps=n, stuck=ex.stuck, out=out), ensure_ascii=False), flush=True)
