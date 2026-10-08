# 10/5版: 冒頭・運び込み=等幅書体の打ち出し / 各稿=機械の声(Claude)の打ち出し / 周回字幕=等幅 / 結び=等幅太字の題名+著作権画面
import os, sys, json, numpy as np
import build_seisho as B
import opening_type as OT
from PIL import Image, ImageDraw
SRC, REC, OUTP = sys.argv[1:4]; FPS = B.FPS; mj, me = OT.mj, OT.me
COPY = "\u00a9 2026 Chitose10. All rights reserved."
def title_runs():
    S = B.S["title"]; y = B.HH/2 - 56
    return [OT.run(B.LX, y, S["ja"], mj(52, 700), B.INK, 10, 13), OT.run(B.LX + 2, y + 84, S["en"], me(18, 500), B.EN_C, 30)]
OT.title_runs = title_runs
def render_on(base, runs, t, i):
    im = base.copy(); d = ImageDraw.Draw(im); clock = 0.0; cur = None; done = True
    for r in runs:
        if r["cps"] is None: OT.draw_part(d, r, r["text"]); continue
        dur = len(r["text"])/r["cps"]
        if t >= clock + dur: cur = (OT.draw_part(d, r, r["text"]), r["y"], r["f"].size); clock += dur; continue
        k = max(0, int((t - clock)*r["cps"])); cur = (OT.draw_part(d, r, r["text"][:k]), r["y"], r["f"].size); done = False; break
    if cur and (not done or (i // 10) % 2 == 0):
        x, y, s = cur; d.rectangle((x + 3, y + 3, x + 3 + int(s*0.55), y + s + 4), fill=B.VERM)
    return im
def draft_base(it, rel, img, final):
    im, d = B.canvas(); pw, ph = 880, 440; x0 = (B.W - pw)//2
    im.paste(Image.fromarray(img).resize((pw, ph), Image.LANCZOS), (x0, 34)); y0 = ph + 34 + 26
    B.spaced(d, x0, y0, f"第{B.KAN[it]}稿", mj(24, 500), B.INK, 6)
    d.text((x0, y0 + 36), f"DRAFT_{['','I','II','III','IV','V'][it]}", font=me(14), fill=B.EN_C)
    d.text((x0, y0 + 62), "  ".join(f"{r:g}\u00b0" for r in rel), font=me(13), fill=B.GREY)
    if final:
        sx, sy = x0 + 118, y0 + 1; d.rectangle((sx, sy, sx + 28, sy + 28), outline=B.VERM, width=2)
        d.text((sx + 14, sy + 14), "定", font=B.jp(18, "Bold"), fill=B.VERM, anchor="mm")
    return im, x0, pw, y0
def draft_runs(quotes, x0, pw, y0):
    qx, qw, y = x0 + 230, pw - 230, y0; Rn = []
    for ja_, en_ in quotes:
        for l in B.wrap_px(OT._d, ja_, mj(16), qw): Rn.append(OT.run(qx, y, l, mj(16), B.INK, 16)); y += 26
        for l in B.wrap_px(OT._d, en_, me(12), qw): Rn.append(OT.run(qx, y, l, me(12), B.EN_C)); y += 18
        y += 9
    Rn.append(OT.run(x0 + pw - OT._d.textlength("— Claude", font=me(12)), min(y, B.HH - 26), "— Claude", me(12), B.GREY))
    return Rn
def carry_runs():
    c = B.S["carry"]; fj, fe = mj(24), me(15); Rn = []
    jl = [x for l in c["ja"] for x in B.wrap_px(OT._d, l, fj, 1000)]; el = B.balanced(OT._d, c["en"], fe, 900)
    y = (B.HH - (len(jl)*44 + 24 + len(el)*26))/2
    for l in jl: Rn.append(OT.run((B.W - OT._d.textlength(l, font=fj))/2, y, l, fj, B.INK, 18)); y += 44
    y += 24
    for l in el: Rn.append(OT.run((B.W - OT._d.textlength(l, font=fe))/2, y, l, fe, B.EN_C)); y += 26
    return Rn
def orbit_overlay(fr, alpha):
    im = Image.fromarray(fr).convert("RGBA"); lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i, a in enumerate(np.linspace(0, 150, 170).astype(np.uint8)): d.line([(0, B.HH - 170 + i), (B.W, B.HH - 170 + i)], fill=B.BG + (int(a*alpha),))
    o = B.S["orbit"]; a8 = int(255*alpha)
    d.text((B.W/2, B.HH - 102), o["ja"], font=mj(21), fill=B.INK + (a8,), anchor="ma")
    d.text((B.W/2, B.HH - 62), o["en"], font=me(14), fill=B.EN_C + (a8,), anchor="ma")
    return np.asarray(Image.alpha_composite(im, lay).convert("RGB"))
def title_end():
    im, d = B.canvas(); S = B.S["title"]; f = mj(52, 700); w = B.spaced_w(d, S["ja"], f, 14)
    B.spaced(d, (B.W - w)/2, B.HH/2 - 60, S["ja"], f, B.INK, 14)
    d.text((B.W/2, B.HH/2 + 34), S["en"], font=me(18, 500), fill=B.EN_C, anchor="ma"); return im
def copyright_screen():
    im, d = B.canvas(); d.text((B.W/2, B.HH/2), COPY, font=me(16), fill=B.GREY, anchor="mm"); return im
if __name__ == "__main__":
    tr = json.load(open(f"{REC}/run02.json", encoding="utf-8"))["trace"]; q = B.Seq(OUTP)
    OT.opening_typed(q, OT.specs(), OT.levels_from(REC)); print("MARK drafts", round(q.n/FPS, 2), flush=True)
    for t in tr:
        img = np.asarray(Image.open(f"{REC}/iter{t['iter']:02d}.png").convert("RGB")); qs = B.S["drafts"][str(t["iter"])]
        base, x0, pw, y0 = draft_base(t["iter"], t["rel_deg"], img, t is tr[-1]); Rn = draft_runs(qs, x0, pw, y0)
        q.fade_to(render_on(base, Rn, -1, 0))
        h = 3.5 + sum(len(a) for a, _ in qs)/12 + (1.5 if t is tr[-1] else 0)
        for i in range(int(FPS*h)): q.put(render_on(base, Rn, i/FPS - 0.2, i))
    print("MARK carry", round(q.n/FPS, 2), flush=True)
    Rn = carry_runs(); q.fade_to(OT.render(Rn, -1, 0))
    for i in range(int(FPS*B.S["carry"]["sec"])): q.put(OT.render(Rn, i/FPS - 0.2, i))
    print("MARK main", round(q.n/FPS, 2), flush=True)
    mf = B.frames(f"{SRC}/main.mp4"); first = next(mf); q.fade_to(first, 1.2); q.put(first)
    for fr in mf: q.put(fr)
    print("MARK orbit", round(q.n/FPS, 2), flush=True)
    of = list(B.frames(f"{SRC}/orbit.mp4")); q.fade_to(of[0], 0.8)
    for i, fr in enumerate(of):
        a = float(np.clip((i - 1.5*FPS)/(1.5*FPS), 0, 1)); q.put(orbit_overlay(fr, a) if a > 0 else fr)
    for _ in range(int(FPS*3.0)): q.put(orbit_overlay(of[-1], 1.0))
    print("MARK ending", round(q.n/FPS, 2), flush=True)
    endbg = of[-1].astype(float)*(1 - B.S["timing"]["veil"]) + np.array(B.BG, float)*B.S["timing"]["veil"]
    q.fade_to(np.clip(endbg*np.asarray(title_end(), float)/np.array(B.BG, float), 0, 255), 1.8); q.hold(5.0)
    print("MARK copyright", round(q.n/FPS, 2), flush=True)
    q.fade_to(copyright_screen(), 1.5); q.hold(2.5); q.fade_to(Image.new("RGB", (B.W, B.HH), B.BG), 1.0); q.hold(0.5)
    q.close(); print("DONE", OUTP, f"{q.n/FPS:.1f}s", flush=True)
