# v14(10/19): 提出物から氏名・アカウント名を除くため、結びの著作権画面の文字だけを差し替える。
#   旧(アカウント名入りの著作権表示、第3引数で与える)→ 新「All rights reserved.」(書体・位置・フェードは同じ)
#   方法: 結びは「題名の静止画面 → 著作権画面(クロスフェード)→ 静止 → 無地(フェード)」なので、各フレームは
#         a×(題名の静止) + b×(著作権画面の静止) + c×(最後の無地) の重ね合わせになる。3つの基準画面はすべて元動画から
#         実測で取り(色・背景の模様・旧文字の形が元動画と完全に一致する)、フレームごとに a・b・c を最小二乗で求めて、
#         b×(新しい著作権画面 − 旧い著作権画面) だけを足す。旧文字が消え、新文字が同じ濃さで現れる。箱の外は一切変えない。
#   使い方: python patch_v14.py <v13の720p版mp4> <出力フォルダ> "<旧著作権表示>" → 置き換えたフレームをPNGで書き出し、開始フレーム番号を表示
import sys, os, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont
W, HH, FPS = 1280, 720, 20
BG, GREY = np.array((239, 237, 232), float), (128, 124, 118)   # 作図上の色(実際の動画では色変換で少しずれる)
FONT = os.environ.get("MONO_FONT", "/content/fonts_mono/JetBrainsMono.ttf")
OLD, NEW = None, "All rights reserved."  # OLDは第3引数で与える
T0, TP = 167.0, 169.0  # T0以降だけを読む / TP=結びの題名が静止している時刻

def screen(text):
    f = ImageFont.truetype(FONT, 16)
    try: f.set_variation_by_axes([400])
    except Exception: pass
    im = Image.new("RGB", (W, HH), tuple(int(v) for v in BG)); ImageDraw.Draw(im).text((W/2, HH/2), text, font=f, fill=GREY, anchor="mm")
    return np.asarray(im, float)

def frames(path, t0):  # 先頭から全部読み、t0より前は捨てる(シークの丸めずれを避ける)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE); k = 0
    while True:
        b = p.stdout.read(W*HH*3)
        if len(b) < W*HH*3: break
        if k >= int(round(t0*FPS)): yield np.frombuffer(b, np.uint8).reshape(HH, W, 3).copy()  # uint8のまま保持(メモリ節約)
        k += 1

def dark(a, mask):  # 文字の濃い側1%の平均色(チャンネル別)
    v = a[mask]; k = max(1, len(v)//100); return np.sort(v, 0)[:k].mean(0)

def patch(src, outdir):
    os.makedirs(outdir, exist_ok=True)
    c_old, c_new = screen(OLD), screen(NEW)
    m = (np.abs(c_old - BG).max(2) > 2) | (np.abs(c_new - BG).max(2) > 2)
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min() - 8, ys.max() + 9, xs.min() - 60, xs.max() + 61  # 書体差を見込んで広めに
    box = lambda f: f[y0:y1, x0:x1].astype(float)
    fs = list(frames(src, T0)); n0 = int(round(T0*FPS))
    avg = lambda a, b: np.mean([box(f) for f in fs[a:b]], 0)                      # 連続フレームの平均(圧縮ノイズを減らす)
    ip = int(round((TP - T0)*FPS)); P, Z = avg(ip - 8, ip + 8), avg(len(fs) - 8, len(fs))  # 題名の静止 / 最後の無地(実測)
    gray = lambda a: np.abs(a - Z).sum(2)
    c0 = next(i for i in range(ip, len(fs)) if np.abs(box(fs[i]) - P).mean() > 1.0)  # 題名→著作権画面のフェード開始
    h = c0 + int(1.5*FPS) + int(0.75*FPS)                                          # フェード1.5秒の後、静止1.5秒の中央
    C = avg(h - 10, h + 10)                                                        # 旧い著作権画面(静止20フレームの平均)
    # 新しい著作権画面: 無地(実測)の上に新文字を、実測の文字色で描く
    tm = np.abs(c_old - BG)[y0:y1, x0:x1].max(2) > 2; Zm = Z.mean((0, 1))
    gn, ga = dark(c_old[y0:y1, x0:x1], tm), dark(C, gray(C) > 6)
    full = Zm + (ga - Zm)*(BG - GREY_ARR)/np.maximum(BG - gn, 1e-6)                # 実測の「文字色」(被覆100%の色)
    cov = ((BG - c_new[y0:y1, x0:x1])/np.maximum(BG - GREY_ARR, 1e-6)).mean(2, keepdims=True)  # 新文字の被覆率(0〜1)
    Cn = Z + cov*(full - Zm)
    from scipy.ndimage import binary_dilation
    M = binary_dilation((gray(C) > 4) | (cov[..., 0] > 0.02), iterations=4)          # 旧文字・新文字の付近
    print(f"静止の検査: 著作権画面の前後フレーム差 {np.abs(box(fs[h]) - box(fs[h + 1])).mean():.2f}(1以下なら静止)")
    A = np.stack([P.ravel(), C.ravel(), Z.ravel()], 1); coef, resid = [], 0.0
    for f in fs:
        d = box(f).ravel(); x = np.linalg.lstsq(A, d, rcond=None)[0]; coef.append(x); resid = max(resid, np.abs(d - A @ x).mean())
    bs = np.array([c[1] for c in coef]); first = int(np.argmax(bs > 0.02))
    print(f"box y{y0}-{y1} x{x0}-{x1} | 文字の出始め frame {n0 + first} ({(n0 + first)/FPS:.2f}s) | 著作権画面の静止 frame {n0 + h} | b最大 {bs.max():.3f}")
    print(f"検査: 重ね合わせの当てはめ誤差(平均) {resid:.2f}(圧縮ノイズ程度=1.5以下なら良) | 実測の文字色 {np.round(full)} 背景 {np.round(Z.mean((0, 1)))}")
    for i, (f, (_, b, _)) in enumerate(zip(fs[first:], coef[first:])):
        g = f.astype(float); a_, b_, c_ = coef[first + i]; gb = g[y0:y1, x0:x1] + min(max(b, 0), 1)*(Cn - C)
        gb[M] = (a_*P + b_*Cn + c_*Z)[M]   # 旧文字・新文字の付近は、重ね合わせの式で描き直す(旧文字の圧縮ノイズの痕を残さない)
        g[y0:y1, x0:x1] = gb
        Image.fromarray(np.clip(np.round(g), 0, 255).astype(np.uint8)).save(f"{outdir}/{i:04d}.png")
    print("書き出し", len(fs) - first, "フレーム")
    return n0 + first

GREY_ARR = np.array(GREY, float)
if __name__ == "__main__":
    OLD = sys.argv[3]
    print("START_FRAME", patch(sys.argv[1], sys.argv[2]))
