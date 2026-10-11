# v13 (10/11): v12 の周回場面の字幕だけを差し替える。ほかのコマ・長さ・音楽は v12 のまま。
# 周回の文字が乗るコマを撮影素材 orbit.mp4 から描き直し、結びへのクロスフェードは差分で補正する。
# 使い方: python3 patch_v13.py <v12_noaudio.mp4> <orbit.mp4> <出力mp4 または 試作の.png> [<フォントのフォルダ>]
import sys, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont
SRC, ORB, OUT = sys.argv[1:4]; FD = sys.argv[4] if len(sys.argv) > 4 else "/content/fonts_mono"
W, HH, FPS = 1280, 720, 20
BG, INK, EN_C = (239, 237, 232), (28, 27, 26), (112, 108, 102)
OLD = ("柱は少しずつ向きを変えながら立ち上がり、ねじれとなって、みずからと環境との差異を刻む。",
       "Turning little by little, the column rises; as a twist, it inscribes the difference between itself and its environment.")
NEW = ("柱は少しずつ向きを変えながら立ち上がり、ねじれとなって、機械の作動と環境との差異を、形として刻む。",
       "Turning little by little, the column rises; its twist inscribes in form the difference between the machine's operation and its environment.")
N0, SKIP, FADE_IN, HOLD, FADE_END = 3069, 96, 16, 40, 36      # v12: MARK orbit 153.45s=3069コマ、ending 165.45s=3309コマ
def font(name, sz):
    f = ImageFont.truetype(f"{FD}/{name}.ttf", sz)
    try: f.set_variation_by_axes([400])
    except Exception: pass
    return f
MJ, ME = font("MPLUS1Code", 21), font("JetBrainsMono", 14)
def overlay(fr, alpha, txt):                       # build_v11.orbit_overlay と同じ描き方
    im = Image.fromarray(fr).convert("RGBA"); lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i, a in enumerate(np.linspace(0, 150, 170).astype(np.uint8)): d.line([(0, HH - 170 + i), (W, HH - 170 + i)], fill=BG + (int(a*alpha),))
    a8 = int(255*alpha)
    d.text((W/2, HH - 102), txt[0], font=MJ, fill=INK + (a8,), anchor="ma")
    d.text((W/2, HH - 62), txt[1], font=ME, fill=EN_C + (a8,), anchor="ma")
    return np.asarray(Image.alpha_composite(im, lay).convert("RGB"))
def frames(path):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W*HH*3)
        if len(b) < W*HH*3: break
        yield np.frombuffer(b, np.uint8).reshape(HH, W, 3)
d0 = ImageDraw.Draw(Image.new("RGB", (8, 8)))
for t, f in ((NEW[0], MJ), (NEW[1], ME)): print("WIDTH", round(d0.textlength(t, font=f)), "/", W, flush=True)
of = list(frames(ORB))[SKIP:]; N1 = N0 + FADE_IN + len(of) + HOLD
print("orbit frames", len(of), "ending at", N1, f"({N1/FPS:.2f}s)", flush=True)
assert N1 == 3309, "v12 の区間と合わない"
if OUT.endswith(".png"): Image.fromarray(overlay(of[-1], 1.0, NEW)).save(OUT); print("PREVIEW", OUT); sys.exit()   # 試作: 1コマだけ描く
alpha = lambda i: float(np.clip((i - 1.5*FPS)/(1.5*FPS), 0, 1))
dL = overlay(of[-1], 1.0, NEW).astype(float) - overlay(of[-1], 1.0, OLD).astype(float)
ts = np.linspace(0, 1, FADE_END)
enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{HH}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", OUT], stdin=subprocess.PIPE)
n = 0; chk = []
for fr in frames(SRC):
    k = n - (N0 + FADE_IN)
    if 0 <= k < len(of):
        a = alpha(k)
        if k in (60, 120): chk.append((k, float(np.abs(fr.astype(float) - overlay(of[k], a, OLD)).mean()))); print("ALIGN", chk[-1], flush=True); assert chk[-1][1] < 8, "位置がずれている"   # 位置合わせの確認
        if a > 0: fr = overlay(of[k], a, NEW)
    elif len(of) <= k < len(of) + HOLD: fr = overlay(of[-1], 1.0, NEW)
    elif 0 <= n - N1 < FADE_END: fr = np.clip(fr.astype(float) + (1 - ts[n - N1])*dL, 0, 255)
    enc.stdin.write(np.asarray(fr, np.uint8).tobytes()); n += 1
enc.stdin.close(); enc.wait()
print("ALIGN", chk, "(小さい値=位置が合っている。目安3以下)", flush=True)
print("DONE", OUT, n, f"{n/FPS:.2f}s", flush=True)
