# 映像用(v12, 10/10): 報告書の図4(循環としての塔)を、映像の表示寸法に合わせて文字を拡大して描き直す。
# 見出しは2行、下端の出典行は省略(出典は映像側に表示)。使い方: python helix_concept_video.py [出力png]
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
OUT = sys.argv[1] if len(sys.argv) > 1 else "helix_cycle_concept_video.png"
fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; fm.fontManager.addfont(fp)
plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
H, R, r = 0.025, 0.16, 0.025; C = np.array([-0.05, 0.0]); cum = np.cumsum([0, 22.5, 22.5, 22.5, 22.5])
picks = np.array([(-0.2478,-0.1707),(-0.0993,-0.2605),(-0.2507,0.1723),(-0.1052,0.2695),(-0.3197,0.0045)]) - C
INK, BLUE, AMB, GREY = "#2c2c2a", "#378add", "#ba7517", "#b4b2a9"
fig = plt.figure(figsize=(7.2, 3.5), dpi=200)
ax = fig.add_subplot(1, 2, 1, projection="3d"); bx = fig.add_subplot(1, 2, 2)
th = np.linspace(0, 2*np.pi, 200)
for ph in np.linspace(0, 2*np.pi, 13)[:-1]:
    ax.plot((R + r*np.cos(ph))*np.cos(th), (R + r*np.cos(ph))*np.sin(th), r + r*np.sin(ph), color=GREY, lw=0.4)
bx.add_patch(plt.Circle((0, 0), (R + r)*1000, fill=False, color=GREY, lw=1)); bx.add_patch(plt.Circle((0, 0), (R - r)*1000, fill=False, color=GREY, lw=1))
bx.text((R+r)*1000*0.72, -(R+r)*1000*0.72, "円環(境界)", ha="left", va="top", fontsize=8, color="#888780")
sq = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*H
for k, a in enumerate(cum):
    t = np.radians(a); Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]); P = sq @ Rm.T
    for z in (2*H*k, 2*H*(k+1)): ax.plot(P[:,0], P[:,1], z, color=INK, lw=0.8)
    for p in P[:4]: ax.plot([p[0]]*2, [p[1]]*2, [2*H*k, 2*H*(k+1)], color=INK, lw=0.8)
zz = np.linspace(0, 10*H, 200); aa = np.interp(zz, [H*(2*k+1) for k in range(5)], cum)
for j in range(4):
    ang = np.radians(45 + 90*j + aa); rc = H*np.sqrt(2)
    ax.plot(rc*np.cos(ang), rc*np.sin(ang), zz, color=BLUE, lw=2)
    bx.plot(rc*np.cos(ang)*1000, rc*np.sin(ang)*1000, color=BLUE, lw=2.5)
for i, p in enumerate(picks):
    s = np.linspace(0, 1, 60); xy = np.outer(1-s, p); zc = 2*H*i + H + 0.14*np.sin(np.pi*s)
    ax.plot(xy[:,0], xy[:,1], zc, color=AMB, lw=1.2, ls="--"); ax.text(p[0], p[1], 0.01, "x", color=AMB, fontsize=10)
    bx.annotate("", xy=(0.12*p[0]*1000/np.linalg.norm(p)*0.3, 0.12*p[1]*1000/np.linalg.norm(p)*0.3), xytext=(p[0]*1000, p[1]*1000),
                arrowprops=dict(arrowstyle="->", color=AMB, lw=1.2, ls="--")); bx.text(p[0]*1000*1.08, p[1]*1000*1.08, "x", color=AMB, fontsize=10, ha="center", va="center")
ax.set_box_aspect((1, 1, 0.5)); ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.3, 0.3); ax.set_zlim(0, 0.3); ax.view_init(18, -60); ax.set_axis_off()
ax.set_title("立面：産出の循環は、\n高さ方向にほどけて螺旋になる", fontsize=10, color=INK)
ins = bx.inset_axes([0.70, 0.56, 0.34, 0.34]); ins.set_aspect("equal"); ins.axis("off")
for j in range(4):
    ang = np.radians(45 + 90*j + aa); ins.plot(np.cos(ang), np.sin(ang), color=BLUE, lw=2)
for k, a in enumerate(cum):
    t = np.radians(45 + a); ins.plot(np.cos(t), np.sin(t), "o", color=INK, ms=3.5)
    ins.text(1.45*np.cos(t), 1.45*np.sin(t), f"$I_{k+1}$" + (" ≡ $I_1$" if k == 4 else ""), ha=("right" if k == 4 else "center"), va="center", fontsize=9, color=INK)
ins.set_xlim(-1.7, 1.7); ins.set_ylim(-1.7, 1.7); ins.text(0, -1.75, "中心部の拡大", ha="center", va="top", fontsize=8, color=INK)
bx.set_aspect("equal"); bx.set_xlim(-330, 330); bx.set_ylim(-330, 330); bx.axis("off")
bx.set_title("平面：4つの角の軌跡は、\n合わせて一周の閉じた円になる", fontsize=10, color=INK)
bx.text(0, -300, "青：石の角の軌跡（各段22.5°、累積90°で一巡）\n橙の破線 x：円環の外から運び込まれる石", ha="center", va="top", fontsize=8.5, color="#5f5e5a", linespacing=1.5)
plt.tight_layout(); plt.savefig(OUT, facecolor="white"); print("ok", OUT)
