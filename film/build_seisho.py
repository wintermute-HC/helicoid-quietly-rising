# 「静かに立ち上がる螺旋体 / Helicoid, Quietly Rising」本編の組み立て
#   文字(screens.json) + 構成記録(run02_re: 予想図とtrace) + 撮影素材(main.mp4, orbit.mp4) → seisho_final.mp4
#   python build_seisho.py <素材フォルダ(main.mp4/orbit.mp4)> <構成記録フォルダ> <出力mp4>
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json, re, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
W, HH, FPS = 1280, 720, 20
SRC, REC, OUTP = (sys.argv[1:4] + ["perf/seisho", "run02_re", "perf/seisho/seisho_final.mp4"][len(sys.argv)-1:])[:3]
S = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "screens.json"), encoding="utf-8"))
FD = "/usr/share/fonts/opentype/noto/"; GF = "/usr/share/fonts/truetype/google-fonts/"
def jp(sz, w="Light"): return ImageFont.truetype(f"{FD}NotoSerifCJK-{w}.ttc", sz)
def en(sz): return ImageFont.truetype(f"{GF}Lora-Italic-Variable.ttf", sz)
def de(sz): return ImageFont.truetype(f"{GF}Lora-Variable.ttf", sz)
BG, INK, GREY, EN_C, VERM = (239, 237, 232), (28, 27, 26), (128, 124, 118), (112, 108, 102), (178, 52, 36)
KAN = "〇一二三四五"
def wrap_px(d, text, f, width):                  # 画素幅で折り返し(和文は1字単位、欧文は語単位、句読点の前では折らない)
    toks = re.findall(r"[A-Za-z0-9'’\",.;:!?()„“”\-–—°/]+\s*|\s+|.", text); out, cur = [], ""
    for t in toks:
        if cur and d.textlength(cur + t.rstrip(), font=f) > width and t not in "、。」』）": out.append(cur.rstrip()); cur = t.lstrip()
        else: cur += t
    return out + ([cur.rstrip()] if cur.strip() else [])
def balanced(d, text, f, width):               # 行数を変えずに各行の長さをそろえる(最終行に1語だけ残さない)
    n = len(wrap_px(d, text, f, width)); w = width
    while w > 200 and len(wrap_px(d, text, f, w - 10)) == n: w -= 10
    return wrap_px(d, text, f, w)
def spaced(d, x, y, text, f, fill, track):
    for c in text: d.text((x, y), c, font=f, fill=fill); x += d.textlength(c, font=f) + track
def spaced_w(d, text, f, track): return sum(d.textlength(c, font=f) + track for c in text) - track
def canvas(): im = Image.new("RGB", (W, HH), BG); return im, ImageDraw.Draw(im)
def title_screen():
    im, d = canvas(); f = jp(46); w = spaced_w(d, S["title"]["ja"], f, 14)
    spaced(d, (W - w)/2, HH/2 - 52, S["title"]["ja"], f, INK, 14)
    g = en(22); d.text((W/2, HH/2 + 38), S["title"]["en"], font=g, fill=EN_C, anchor="ma"); return im
def epigraph_screen():
    im, d = canvas(); x, width = 200, 880; y = 250; e = S["epigraph"]
    for l in wrap_px(d, e["de"], de(25), width): d.text((x, y), l, font=de(25), fill=INK); y += 40
    y += 26
    for l in e["ja"]: d.text((x, y), l, font=jp(22), fill=INK); y += 38
    d.text((x + width, y + 26), e["cite"], font=en(16), fill=GREY, anchor="ra"); return im
def text_screen(item):
    im, d = canvas(); big = item.get("big"); fj = jp(40 if big else 26); fe = en(20 if big else 17)
    jl = item["ja"]; el = balanced(d, item["en"], fe, 900)
    lh_j, lh_e = (60 if big else 46), 28; tot = len(jl)*lh_j + 24 + len(el)*lh_e; y = (HH - tot)/2
    for l in jl:
        if big: w = spaced_w(d, l, fj, 12); spaced(d, (W - w)/2, y, l, fj, INK, 12)
        else: d.text((W/2, y), l, font=fj, fill=INK, anchor="ma")
        y += lh_j
    y += 24
    for l in el: d.text((W/2, y), l, font=fe, fill=EN_C, anchor="ma"); y += lh_e
    return im
LX, LW = 110, 600                                # 冒頭の文字は左の列に
def title_left():
    im, d = canvas(); f = jp(44); spaced(d, LX, HH/2 - 50, S["title"]["ja"], f, INK, 13)
    d.text((LX + 2, HH/2 + 26), S["title"]["en"], font=en(21), fill=EN_C); return im
def epigraph_left():
    im, d = canvas(); e = S["epigraph"]; y = 220
    for l in wrap_px(d, e["de"], de(22), LW): d.text((LX, y), l, font=de(22), fill=INK); y += 34
    y += 22
    for l in e["ja"]: d.text((LX, y), l, font=jp(20), fill=INK); y += 34
    d.text((LX, y + 22), e["cite"], font=en(15), fill=GREY); return im
def text_left(item):
    im, d = canvas(); big = item.get("big"); fj = jp(38 if big else 24); fe = en(19 if big else 16)
    jl = [x for l in item["ja"] for x in wrap_px(d, l, fj, LW)]; el = balanced(d, item["en"], fe, LW)
    lh_j = 58 if big else 42; tot = len(jl)*lh_j + 22 + len(el)*26; y = (HH - tot)/2
    for l in jl:
        if big: spaced(d, LX, y, l, fj, INK, 12)
        else: d.text((LX, y), l, font=fj, fill=INK)
        y += lh_j
    y += 22
    for l in el: d.text((LX, y), l, font=fe, fill=EN_C); y += 26
    return im
def draft_card(it, rel, img, final, quotes):
    im, d = canvas(); pw, ph = 880, 440; x0 = (W - pw)//2
    im.paste(Image.fromarray(img).resize((pw, ph), Image.LANCZOS), (x0, 34))
    y0 = ph + 34 + 26
    spaced(d, x0, y0, f"第{KAN[it]}稿", jp(24, "Medium"), INK, 8)
    d.text((x0, y0 + 36), f"Draft {['','I','II','III','IV','V'][it]}", font=en(15), fill=EN_C)
    d.text((x0, y0 + 62), "  ".join(f"{r:g}°" for r in rel), font=jp(13), fill=GREY)
    if final:
        sx, sy = x0 + 118, y0 + 1; d.rectangle((sx, sy, sx + 28, sy + 28), outline=VERM, width=2)
        d.text((sx + 14, sy + 14), "定", font=jp(18, "Bold"), fill=VERM, anchor="mm")
    qx, qw, y = x0 + 230, pw - 230, y0
    for ja_, en_ in quotes:
        for l in wrap_px(d, ja_, jp(18), qw): d.text((qx, y), l, font=jp(18), fill=INK); y += 27
        for l in balanced(d, en_, en(13), qw): d.text((qx, y), l, font=en(13), fill=EN_C); y += 19
        y += 9
    d.text((x0 + pw, min(y, HH - 26)), "— Claude", font=en(13), fill=GREY, anchor="ra"); return im
class Seq:
    def __init__(s, path):
        s.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{HH}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path], stdin=subprocess.PIPE); s.last = np.full((HH, W, 3), BG, float); s.n = 0
    def put(s, fr): s.p.stdin.write(np.asarray(fr, np.uint8).tobytes()); s.last = np.asarray(fr, float); s.n += 1
    def fade_to(s, im, sec=0.9):
        b = np.asarray(im, float)
        for t in np.linspace(0, 1, int(FPS*sec)): s.put(s.last*(1-t) + b*t)
    def hold(s, sec):
        for _ in range(int(FPS*sec)): s.put(s.last)
    def close(s): s.p.stdin.close(); s.p.wait()
def frames(path):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W*HH*3)
        if len(b) < W*HH*3: break
        yield np.frombuffer(b, np.uint8).reshape(HH, W, 3)
def orbit_overlay(fr, alpha):                    # 周回の映像の下部に、文字を静かに重ねる
    im = Image.fromarray(fr).convert("RGBA"); lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    band = np.linspace(0, 150, 170).astype(np.uint8)                              # 下端へ向かう淡い地のぼかし
    for i, a in enumerate(band): d.line([(0, HH - 170 + i), (W, HH - 170 + i)], fill=BG + (int(a*alpha),))
    o = S["orbit"]; a8 = int(255*alpha)
    d.text((W/2, HH - 102), o["ja"], font=jp(22), fill=INK + (a8,), anchor="ma")
    d.text((W/2, HH - 62), o["en"], font=en(16), fill=EN_C + (a8,), anchor="ma")
    return np.asarray(Image.alpha_composite(im, lay).convert("RGB"))
def backdrop(levels, n):                         # 冒頭の背景: 完成予想図(ロボットなし)の塔を、低い位置からゆっくり回り込みながら見上げる
    import mujoco
    from preview import _xml
    from tower_env import RING_C, TABLE_Z, lookat_quat
    m = mujoco.MjModel.from_xml_string(_xml(levels).replace('offwidth="640" offheight="640"', f'offwidth="{W}" offheight="{HH}"'))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, HH, W); cam = m.camera("art").id
    tgt = np.array([RING_C[0], RING_C[1], TABLE_Z + 0.13])
    for i in range(n):
        s_ = i/max(n-1, 1); th = np.radians(-75 + 95*s_); rad = 0.62 - 0.12*s_; z = TABLE_Z + 0.05 + 0.14*s_
        pos = np.r_[tgt[:2] + rad*np.array([np.cos(th), np.sin(th)]), z]
        fwd = tgt - pos; right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
        aim = tgt - right*0.16*rad/0.6                                           # 塔を画面の右三分の一に置く(左に文字)
        m.cam_pos[cam] = pos; m.cam_quat[cam] = lookat_quat(pos, aim); mujoco.mj_forward(m, d)
        r.update_scene(d, camera="art"); yield r.render().copy()
def opening(q, screens, levels):                 # 文字の画面を、動き続ける予想図の上に重ねる(乗算合成: 地の色は透け、文字だけが残る)
    T = S["timing"]; veil = T["veil"]; fade = int(FPS*0.9)
    plan = []                                                                   # (文字画面, 保持秒)
    for im, sec in screens: plan.append((np.asarray(im, float)/np.array(BG, float), sec))
    n = sum(fade + int(FPS*sec) for _, sec in plan) + fade; bg = backdrop(levels, n); prev = np.ones((HH, W, 3))
    Bf = np.array(BG, float)
    def emit(layer, k):
        f = next(bg).astype(float)*(1 - veil) + Bf*veil
        if k is not None: f = f*(1 - k) + Bf*k                                  # 冒頭は地の色から立ち上げる
        q.put(np.clip(f*layer, 0, 255))
    for j, (lay, sec) in enumerate(plan):
        for t in np.linspace(0, 1, fade): emit(prev*(1-t) + lay*t, (1 - t) if j == 0 else None)
        for _ in range(int(FPS*sec)): emit(lay, None)
        prev = lay
    last = None
    for t in np.linspace(0, 1, fade): emit(prev*(1-t) + 1.0*t, t)          # 文字と背景を地の色へ溶かす
    q.last = np.tile(Bf, (HH, W, 1))
if __name__ == "__main__":
    log = json.load(open(f"{REC}/run02.json", encoding="utf-8")); tr = log["trace"]; T = S["timing"]
    fin = [dict(color=c, rel_deg=r, dx_mm=0, dy_mm=0) for c, r in zip("BWBWB", tr[-1]["rel_deg"])]
    q = Seq(OUTP)
    opening(q, [(title_left(), T["title"]), (epigraph_left(), T["epigraph"])] + [(text_left(it), it["sec"]) for it in S["before"]], fin)
    for t in tr:
        img = np.asarray(Image.open(f"{REC}/iter{t['iter']:02d}.png").convert("RGB")); qs = S["drafts"][str(t["iter"])]
        q.fade_to(draft_card(t["iter"], t["rel_deg"], img, t is tr[-1], qs))
        q.hold(3.5 + sum(len(a) for a, _ in qs)/12 + (1.5 if t is tr[-1] else 0))
    c = S["carry"]; q.fade_to(text_screen(c)); q.hold(c["sec"])
    mf = frames(f"{SRC}/main.mp4"); first = next(mf); q.fade_to(first, 1.2); q.put(first)
    for fr in mf: q.put(fr)
    of = list(frames(f"{SRC}/orbit.mp4")); q.fade_to(of[0], 0.8); n = len(of)
    for i, fr in enumerate(of):
        a = float(np.clip((i - 1.5*FPS)/(1.5*FPS), 0, 1)); q.put(orbit_overlay(fr, a) if a > 0 else fr)
    for _ in range(int(FPS*3.0)): q.put(orbit_overlay(of[-1], 1.0))
    endbg = of[-1].astype(float)*(1 - S["timing"]["veil"]) + np.array(BG, float)*S["timing"]["veil"]   # 結びの題名は、完成した塔の上に
    q.fade_to(np.clip(endbg*np.asarray(title_screen(), float)/np.array(BG, float), 0, 255), 1.8); q.hold(5.0)
    q.fade_to(Image.new("RGB", (W, HH), BG), 1.5); q.hold(0.5)
    q.close(); print("DONE", OUTP, f"{q.n/FPS:.1f}s")
