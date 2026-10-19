# 作品の一本化: 言葉 → Claudeが構成(予想図を見て自己批評・修正) → 見て積むロボットが実行・撮影 → 1本の映像に
#   python perform.py --words "静かに立ち上がる螺旋" --out perf/seisho      (Claude APIで構成。ANTHROPIC_API_KEYが必要)
#   python perform.py --replay compose/run02 --out perf/seisho              (保存済みの構成記録を再生して撮影だけ行う)
# 出力: composing.mp4(構成過程) main.mp4(組み上げ) orbit.mp4(周回) full.mp4(3本を連結) plan.json report.json
import os, sys, json, glob, argparse, subprocess, textwrap; os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
W, HH, FPS = 1280, 720, 20
FD = "/usr/share/fonts/opentype/noto/"
def font(sz, w="Regular"):
    for f in (f"{FD}NotoSerifCJK-{w}.ttc", f"{FD}NotoSerifCJK-Regular.ttc", f"{FD}NotoSansCJK-Regular.ttc"):
        if os.path.exists(f): return ImageFont.truetype(f, sz)
    return ImageFont.load_default()
BG, INK, GREY, VERM = (239, 237, 232), (28, 27, 26), (128, 124, 118), (178, 52, 36)
KAN = "〇一二三四五六七八九十"
def wrap(t, n):                                   # 日本語の折り返し(数値・角度の途中と、句読点・閉じ括弧の直前では折らない)
    import re; out, cur = [], ""
    for tok in re.findall(r"-?\d+(?:\.\d+)?°?|[A-Za-z]+|.", t):
        if len(cur) + len(tok) > n and tok not in "、。」』）": out.append(cur); cur = ""
        cur += tok
    return out + ([cur] if cur else [])
def spaced(d, xy, text, f, fill, track=6, anchor_right=False):   # 字間を空けて描く
    w = sum(d.textlength(c, font=f) + track for c in text) - track; x, y = xy
    if anchor_right: x -= w
    for c in text: d.text((x, y), c, font=f, fill=fill); x += d.textlength(c, font=f) + track
def title_card(words):
    """冒頭: 言葉だけを置く"""
    im = Image.new("RGB", (W, HH), BG); d = ImageDraw.Draw(im); f = font(46, "Light")
    w = sum(d.textlength(c, font=f) + 14 for c in words) - 14
    spaced(d, ((W - w)/2, HH/2 - 40), words, f, INK, 14)
    return np.asarray(im)
def card(words, it, n, tr, img, final):
    """一稿ごと: 予想図、稿の番号と回転の並び、Claudeの言葉の引用"""
    im = Image.new("RGB", (W, HH), BG); d = ImageDraw.Draw(im)
    pv = Image.fromarray(img).resize((960, 480), Image.LANCZOS); im.paste(pv, ((W - 960)//2, 48))
    x0, y0 = (W - 960)//2, 562
    spaced(d, (x0, y0), f"第{KAN[it]}稿", font(26, "Medium"), INK, 8)
    spaced(d, (x0, y0 + 46), "  ".join(f"{r:g}°" for r in tr["rel_deg"]), font(15, "Light"), GREY, 2)
    if final:                                                                 # 朱の印「定」
        sx, sy = x0 + 128, y0 + 2; d.rectangle((sx, sy, sx + 30, sy + 30), outline=VERM, width=2)
        d.text((sx + 15, sy + 15), "定", font=font(20, "Bold"), fill=VERM, anchor="mm")
    cap = tr.get("caption") or [tr.get("critique") or tr.get("concept") or ""]
    y = y0; qx = x0 + 250; f = font(20, "Light")
    for sent in cap:
        for l in wrap(sent, 34): d.text((qx, y), l, font=f, fill=INK); y += 31
        y += 8
    spaced(d, (x0 + 960, y + 2), "— Claude", font(14, "Light"), GREY, 2, anchor_right=True)
    return np.asarray(im)
def ffmpeg_writer(path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{HH}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path], stdin=subprocess.PIPE)
def composing_video(words, trace, img_dir, path, hold=3.5):
    p = ffmpeg_writer(path); blank = np.full((HH, W, 3), BG, np.uint8).astype(float)
    def fade(a_, b_, sec=0.8):
        for t in np.linspace(0, 1, int(FPS*sec)): p.stdin.write((a_*(1-t) + b_*t).astype(np.uint8).tobytes())
    def still(fr, sec):
        for _ in range(int(FPS*sec)): p.stdin.write(fr.astype(np.uint8).tobytes())
    tc = title_card(words).astype(float); fade(blank, tc, 1.2); still(tc, 3.0); prev = tc
    for tr in trace:
        img = np.asarray(Image.open(f"{img_dir}/iter{tr['iter']:02d}.png").convert("RGB"))
        fr = card(words, tr["iter"], len(trace), tr, img, tr is trace[-1]).astype(float)
        fade(prev, fr); nchar = sum(len(x) for x in (tr.get("caption") or []))
        still(fr, hold + nchar/12 + (2.0 if tr is trace[-1] else 0)); prev = fr
    fade(prev, blank, 1.0); p.stdin.close(); p.wait()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--words"); ap.add_argument("--replay"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=700); ap.add_argument("--max_iter", type=int, default=5); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); img_dir = f"{a.out}/compose"
    if a.replay:                                                           # 保存済みの構成記録(json + iterXX.png)
        js = [j for j in glob.glob(f"{a.replay}/*.json") if '"trace"' in open(j, encoding="utf-8").read()]
        log = json.load(open(js[0], encoding="utf-8")); words, trace = log["words"], log["trace"]; img_dir = a.replay
        cap = f"{a.replay}/captions.json"
        if os.path.exists(cap):
            c = json.load(open(cap, encoding="utf-8")); [tr.update(caption=c.get(str(tr["iter"]))) for tr in trace]
    else:
        from composer import compose
        words = a.words; _, trace = compose(words, max_iter=a.max_iter, log_path=f"{a.out}/compose_log.json", img_dir=img_dir)
    fin = trace[-1]; rel = fin["rel_deg"]
    json.dump(dict(words=words, title=fin.get("title"), concept=fin.get("concept"), rel_deg=rel, iterations=len(trace), final=fin.get("final")),
              open(f"{a.out}/plan.json", "w"), ensure_ascii=False, indent=1)
    print(f"構成確定: {fin.get('title')} 回転 {rel}（{len(trace)}回）", flush=True)
    composing_video(words, trace, img_dir, f"{a.out}/composing.mp4")
    env = dict(os.environ, REL=",".join(str(r) for r in rel), SEED=str(a.seed), OUT=a.out)
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "archive", "film.py")], env=env, capture_output=True, text=True)
    print(*[l for l in r.stdout.splitlines() if l.startswith("{")], flush=True)
    lst = f"{a.out}/concat.txt"; open(lst, "w").write("".join(f"file '{os.path.abspath(f'{a.out}/{v}.mp4')}'\n" for v in ("composing", "main", "orbit")))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", f"{a.out}/full.mp4"])
    print("DONE", a.out, flush=True)
if __name__ == "__main__": main()
