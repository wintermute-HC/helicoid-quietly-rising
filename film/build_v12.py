# 10/10版(build_v12): 最終版の映像組み立て(175.2秒)。build_v11 の部品を流用し、次を変更:
#   引用の次にコンセプト画面(引用+河本図+循環としての塔+コンセプト文) / 第一〜第四稿を2×2の1枚に / 周回の冒頭を省略
# 使い方: MARKS=marks1009.json HELICOID_FIG=<図のフォルダ> python build_v12.py <撮影素材フォルダ> <構成記録フォルダ> <出力mp4>
#         python build_v12.py --preview   (コンセプト画面と2×2シートの1コマを /content/v12_prev に書き出す)
# 図のフォルダには次の2点を置く:
#   kawamoto1993_p46_fig_crop.png  河本英夫『現代思想』1993年9月号 p.46 の原図の写真(著作物のため本リポジトリには含めない)
#   helix_cycle_concept_video.png  循環としての塔の図(helix_concept_video.py で作図)
import os, sys, json, numpy as np
from PIL import Image, ImageDraw
PREVIEW = len(sys.argv) > 1 and sys.argv[1] == "--preview"
if PREVIEW: sys.argv = [sys.argv[0], "x", "/content/rec", "x.mp4"]
import build_seisho as B
import opening_type as OT
import build_v11 as V
SRC, REC, OUTP = sys.argv[1:4]; FPS = B.FPS; mj, me = OT.mj, OT.me
FIG = os.environ.get("HELICOID_FIG", "/content/fig")
def fig_layout(H=340, gap=48, y0=120):
    k = Image.open(f"{FIG}/kawamoto1993_p46_fig_crop.png").convert("RGB")
    h = Image.open(f"{FIG}/helix_cycle_concept_video.png").convert("RGB")
    k = k.resize((round(k.width*H/k.height), H), Image.LANCZOS); h = h.resize((round(h.width*H/h.height), H), Image.LANCZOS)
    h = Image.fromarray((np.asarray(h, float)*np.array(B.BG, float)/255).astype(np.uint8))   # 自作図の白地を映像の地の色に
    return (B.W - (k.width + gap + h.width))//2, y0, k, h, gap, H
FL = fig_layout()
def fig_pack():                                   # 図の画素と、図を不透明に置く範囲(枠を含む)
    x0, y0, k, h, gap, H = FL; A = np.tile(np.array(B.BG, float), (B.HH, B.W, 1)); M = np.zeros((B.HH, B.W, 1))
    A[y0-1:y0+H+1, x0-1:x0+k.width+1] = B.GREY; A[y0:y0+H, x0:x0+k.width] = np.asarray(k); M[y0-1:y0+H+1, x0-1:x0+k.width+1] = 1
    hx = x0 + k.width + gap; A[y0:y0+H, hx:hx+h.width] = np.asarray(h); M[y0:y0+H, hx:hx+h.width] = 1
    return A, M
FIGS = fig_pack()
def concept_runs():
    e, c = B.S["epigraph"], B.S["concept"]; x0, y0, k, h, gap, H = FL; R = []
    R += [OT.run(x0, 34, e["de"], me(13), B.INK), OT.run(x0, 58, "".join(e["ja"]), mj(15), B.INK), OT.run(x0, 84, e["cite"], me(11), B.GREY)]
    yc = y0 + H + 10; ym = yc
    for cap, cx, cw in [(c["fig_left"], x0, k.width), (c["fig_right"], x0 + k.width + gap, h.width)]:
        y = yc
        for l in [x for s_ in (cap["ja"] if isinstance(cap["ja"], list) else [cap["ja"]]) for x in B.wrap_px(OT._d, s_, mj(11), cw)]: R.append(OT.run(cx, y, l, mj(11), B.GREY)); y += 16
        for l in B.wrap_px(OT._d, cap["en"], me(10), cw): R.append(OT.run(cx, y, l, me(10), B.GREY)); y += 14
        ym = max(ym, y)
    y = ym + 22; TW = k.width + gap + h.width
    for l in c["ja"]: R.append(OT.run(x0, y, l, mj(19), B.INK, 18)); y += 32
    y += 10
    for l in B.balanced(OT._d, c["en"], me(13), TW): R.append(OT.run(x0, y, l, me(13), B.EN_C)); y += 21
    return R
def specs12():
    T, c = B.S["timing"], B.S["concept"]
    return ([(OT.title_runs(), T["title"], 0.3, False), (OT.epigraph_runs(), T["epigraph"], 0.3, False),
             (concept_runs(), c["sec"], c["delay"], True)] + [(OT.text_runs(it), it["sec"], 0.3, False) for it in B.S["before"]])
def opening_typed12(q, specs, levels):            # v11の冒頭と同じ合成。図の画面だけ背後の塔を地の色に溶かし、図を不透明に置く
    veil = B.S["timing"]["veil"]; fade = int(FPS*0.9); Bf = np.array(B.BG, float); A, M = FIGS
    n = sum(fade + int(FPS*s[1]) for s in specs) + fade; bg = B.backdrop(levels, n)
    def emit(layer, k, a):
        vv = veil + (1 - veil)*a; f = next(bg).astype(float)*(1 - vv) + Bf*vv
        if k is not None: f = f*(1 - k) + Bf*k
        out = np.clip(f*layer, 0, 255)
        if a > 0: out = out*(1 - a*M) + A*(a*M)
        q.put(out)
    prev = np.ones((B.HH, B.W, 3)); pa = 0.0
    for j, (runs, sec, delay, fig) in enumerate(specs):
        a1 = 1.0 if fig else 0.0; tgt = np.asarray(OT.render(runs, -1, 1), float)/Bf if fig else 1.0
        for t in np.linspace(0, 1, fade): emit(prev*(1 - t) + tgt*t, (1 - t) if j == 0 else None, pa*(1 - t) + a1*t)
        for i in range(int(FPS*sec)): lay = np.asarray(OT.render(runs, i/FPS - delay, i), float)/Bf; emit(lay, None, a1)
        prev = lay; pa = a1
    for t in np.linspace(0, 1, fade): emit(prev*(1 - t) + 1.0*t, t, pa*(1 - t))
    q.last = np.tile(Bf, (B.HH, B.W, 1))
def sheet_base(items):                            # 第一〜第四稿を2×2に
    im, d = B.canvas(); iw, ih, gx = 400, 200, 80; X0 = (B.W - 2*iw - gx)//2; cells = []
    for n, (it, rel, img) in enumerate(items):
        x = X0 + (n % 2)*(iw + gx); y = 26 + (n//2)*330
        im.paste(Image.fromarray(img).resize((iw, ih), Image.LANCZOS), (x, y)); ty = y + ih + 10
        B.spaced(d, x, ty, f"第{B.KAN[it]}稿", mj(16, 500), B.INK, 4)
        d.text((x + 84, ty + 4), f"DRAFT_{['','I','II','III','IV','V'][it]}", font=me(11), fill=B.EN_C)
        d.text((x + 170, ty + 4), "  ".join(f"{r:g}°" for r in rel), font=me(11), fill=B.GREY)
        cells.append((x, ty + 30))
    return im, cells, X0, 2*iw + gx
def sheet_runs(qss, cells, X0, TW):
    Rn = []; yb = 0
    for qs, (x, y) in zip(qss, cells):
        for ja_, en_ in qs:
            for l in (ja_.split("\n") if "\n" in ja_ else B.wrap_px(OT._d, ja_, mj(15), 400)): Rn.append(OT.run(x, y, l, mj(15), B.INK, 16)); y += 22
            y += 3
            for l in B.wrap_px(OT._d, en_, me(11), 400): Rn.append(OT.run(x, y, l, me(11), B.EN_C)); y += 16
        yb = max(yb, y)
    Rn.append(OT.run(X0 + TW - OT._d.textlength("— Claude", font=me(12)), min(yb + 4, B.HH - 22), "— Claude", me(12), B.GREY))
    return Rn
def load_sheet(tr):                               # 和文の改行位置は screens.json の sheet_ja(句読点で改行、文面は原文のまま)
    items = [(t["iter"], t["rel_deg"], np.asarray(Image.open(f"{REC}/iter{t['iter']:02d}.png").convert("RGB"))) for t in tr[:-1]]
    qss = [[(B.S.get("sheet_ja", {}).get(str(t["iter"]), a), e) for a, e in B.S["drafts"][str(t["iter"])]] for t in tr[:-1]]; base, cells, X0, TW = sheet_base(items)
    return base, sheet_runs(qss, cells, X0, TW), 2.0 + sum(len(a) for qs in qss for a, _ in qs)/16
if PREVIEW:
    os.makedirs("/content/v12_prev", exist_ok=True); A, M = FIGS
    im = np.asarray(OT.render(concept_runs(), 99, 10), float)
    Image.fromarray(np.clip(im*(1 - M) + A*M, 0, 255).astype(np.uint8)).save("/content/v12_prev/concept.png")
    tr = json.load(open(f"{REC}/run02.json", encoding="utf-8"))["trace"]; base, Rn, hs = load_sheet(tr)
    V.render_on(base, Rn, 99, 10).save("/content/v12_prev/sheet.png")
    print(f"EST concept {0.9 + B.S['concept']['sec']:.1f}s / sheet {0.9 + hs:.1f}s (v11 第一〜四稿 20.05s) / orbit -{B.S['timing']['orbit_skip']}s")
elif __name__ == "__main__":
    tr = json.load(open(f"{REC}/run02.json", encoding="utf-8"))["trace"]; assert len(tr) == 5; q = B.Seq(OUTP)
    opening_typed12(q, specs12(), OT.levels_from(REC)); print("MARK drafts", round(q.n/FPS, 2), flush=True)
    base, Rn, hs = load_sheet(tr); q.fade_to(V.render_on(base, Rn, -1, 0))
    for i in range(int(FPS*hs)): q.put(V.render_on(base, Rn, i/FPS - 0.2, i))
    t = tr[-1]; print("MARK d5", round(q.n/FPS, 2), flush=True)
    img = np.asarray(Image.open(f"{REC}/iter{t['iter']:02d}.png").convert("RGB")); qs = B.S["drafts"][str(t["iter"])]
    base, x0, pw, y0 = V.draft_base(t["iter"], t["rel_deg"], img, True); Rn = V.draft_runs(qs, x0, pw, y0)
    q.fade_to(V.render_on(base, Rn, -1, 0)); h = 2.0 + sum(len(a) for a, _ in qs)/16 + 1.5
    for i in range(int(FPS*h)): q.put(V.render_on(base, Rn, i/FPS - 0.2, i))
    print("MARK carry", round(q.n/FPS, 2), flush=True)
    Rn = V.carry_runs(); q.fade_to(OT.render(Rn, -1, 0))
    for i in range(int(FPS*B.S["carry"]["sec"])): q.put(OT.render(Rn, i/FPS - 0.2, i))
    print("MARK main", round(q.n/FPS, 2), flush=True)
    mf = B.frames(f"{SRC}/main.mp4"); first = next(mf); q.fade_to(first, 1.2); q.put(first)
    MK = json.load(open(os.environ.get('MARKS', '/content/marks1009.json'))); FA, FB = MK['fa'], MK['fb']   # 2〜4個目の石(marks1009.py)を2倍速
    for idx, fr in enumerate(mf, start=1):
        if FA <= idx < FB:
            if idx % 2: continue
            fr = V.speed_badge(fr)
        q.put(fr)
    print("MARK orbit", round(q.n/FPS, 2), flush=True)
    of = list(B.frames(f"{SRC}/orbit.mp4"))[int(FPS*B.S["timing"]["orbit_skip"]):]; q.fade_to(of[0], 0.8)
    for i, fr in enumerate(of):
        a = float(np.clip((i - 1.5*FPS)/(1.5*FPS), 0, 1)); q.put(V.orbit_overlay(fr, a) if a > 0 else fr)
    for _ in range(int(FPS*2.0)): q.put(V.orbit_overlay(of[-1], 1.0))
    print("MARK ending", round(q.n/FPS, 2), flush=True)
    endbg = of[-1].astype(float)*(1 - B.S["timing"]["veil"]) + np.array(B.BG, float)*B.S["timing"]["veil"]
    q.fade_to(np.clip(endbg*np.asarray(V.title_end(), float)/np.array(B.BG, float), 0, 255), 1.8); q.hold(3.5)
    print("MARK copyright", round(q.n/FPS, 2), flush=True)
    q.fade_to(V.copyright_screen(), 1.5); q.hold(1.5); q.fade_to(Image.new("RGB", (B.W, B.HH), B.BG), 1.0); q.hold(0.5)
    q.close(); print("DONE", OUTP, f"{q.n/FPS:.1f}s", flush=True)
