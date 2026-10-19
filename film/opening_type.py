# 冒頭(題名・引用・台座・指示文)を等幅書体の打ち出しで描く(10/5試作)。画面の秒数・総コマ数は build_seisho と同じ
import sys, json, functools, numpy as np
import build_seisho as B
from PIL import Image, ImageDraw, ImageFont
FD = "/content/fonts_mono"
@functools.lru_cache(None)
def var(path, sz, w):
    f = ImageFont.truetype(path, sz)
    try: f.set_variation_by_axes([w])
    except Exception: pass
    return f
mj = lambda sz, w=400: var(f"{FD}/MPLUS1Code.ttf", sz, w)
me = lambda sz, w=400: var(f"{FD}/JetBrainsMono.ttf", sz, w)
_d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
def run(x, y, text, f, fill, cps=None, track=0): return dict(x=x, y=y, text=text, f=f, fill=fill, cps=cps, track=track)
def title_runs():
    S = B.S["title"]; y = B.HH/2 - 50
    return [run(B.LX, y, S["ja"], mj(42, 300), B.INK, 10, 13), run(B.LX + 2, y + 74, S["en"], me(18), B.EN_C, 30)]
def epigraph_runs():
    e = B.S["epigraph"]; y = 220; R = []
    for l in B.wrap_px(_d, e["de"], me(18), B.LW): R.append(run(B.LX, y, l, me(18), B.INK, 45)); y += 30
    y += 22
    for l in e["ja"]: R.append(run(B.LX, y, l, mj(19), B.INK, 18)); y += 34
    R.append(run(B.LX, y + 22, e["cite"], me(13), B.GREY)); return R
def text_runs(item):
    big = item.get("big"); fj = mj(36, 300) if big else mj(22); fe = me(17) if big else me(14)
    jl = [x for l in item["ja"] for x in B.wrap_px(_d, l, fj, B.LW)]; el = B.balanced(_d, item["en"], fe, B.LW)
    lh = 58 if big else 40; tot = len(jl)*lh + 22 + len(el)*24; y = (B.HH - tot)/2; R = []
    for l in jl: R.append(run(B.LX, y, l, fj, B.INK, 8 if big else 18, 12 if big else 0)); y += lh
    y += 22
    for l in el: R.append(run(B.LX, y, l, fe, B.EN_C)); y += 24
    return R
def draw_part(d, r, s):
    if r["track"]: B.spaced(d, r["x"], r["y"], s, r["f"], r["fill"], r["track"]); return r["x"] + (B.spaced_w(d, s, r["f"], r["track"]) + r["track"] if s else 0)
    d.text((r["x"], r["y"]), s, font=r["f"], fill=r["fill"]); return r["x"] + d.textlength(s, font=r["f"])
def render(runs, t, i):
    im, d = B.canvas(); clock = 0.0; cur = None; done = True
    for r in runs:
        if r["cps"] is None: draw_part(d, r, r["text"]); continue
        dur = len(r["text"])/r["cps"]
        if t >= clock + dur: cur = (draw_part(d, r, r["text"]), r["y"], r["f"].size); clock += dur; continue
        k = max(0, int((t - clock)*r["cps"])); cur = (draw_part(d, r, r["text"][:k]), r["y"], r["f"].size); done = False; break
    if cur and (not done or (i // 10) % 2 == 0):
        x, y, s = cur; d.rectangle((x + 3, y + 3, x + 3 + int(s*0.55), y + s + 4), fill=B.VERM)
    return im
def opening_typed(q, specs, levels):
    FPS = B.FPS; veil = B.S["timing"]["veil"]; fade = int(FPS*0.9); Bf = np.array(B.BG, float)
    n = sum(fade + int(FPS*sec) for _, sec in specs) + fade; bg = B.backdrop(levels, n); prev = np.ones((B.HH, B.W, 3))
    def emit(layer, k):
        f = next(bg).astype(float)*(1 - veil) + Bf*veil
        if k is not None: f = f*(1 - k) + Bf*k
        q.put(np.clip(f*layer, 0, 255))
    for j, (runs, sec) in enumerate(specs):
        for t in np.linspace(0, 1, fade): emit(prev*(1 - t) + 1.0*t, (1 - t) if j == 0 else None)
        for i in range(int(FPS*sec)):
            lay = np.asarray(render(runs, i/FPS - 0.3, i), float)/Bf; emit(lay, None)
        prev = lay
    for t in np.linspace(0, 1, fade): emit(prev*(1 - t) + 1.0*t, t)
    q.last = np.tile(Bf, (B.HH, B.W, 1))
def specs():
    T = B.S["timing"]
    return [(title_runs(), T["title"]), (epigraph_runs(), T["epigraph"])] + [(text_runs(it), it["sec"]) for it in B.S["before"]]
def levels_from(rec):
    tr = json.load(open(f"{rec}/run02.json", encoding="utf-8"))["trace"]
    return [dict(color=c, rel_deg=r, dx_mm=0, dy_mm=0) for c, r in zip("BWBWB", tr[-1]["rel_deg"])]
if __name__ == "__main__":
    q = B.Seq(sys.argv[1]); opening_typed(q, specs(), levels_from("/content/rec")); q.close(); print("DONE", f"{q.n/B.FPS:.1f}s")
