# 知覚: 真上の深度カメラ画像から、石(立方体)の位置・向き・色と、塔の最上段を推定する(シミュレータの正解は使わない)
import numpy as np
from scipy import ndimage
from tower_env import RING_C, RING_R, RING_r, TABLE_Z, H, TOPCAM_Z, TOPCAM_FOVY
RES = 512
def capture(env):
    rgb, d = env.sim.render(width=RES, height=RES, camera_name="topcam", depth=True)
    m = env.sim.model; ext = m.stat.extent; near, far = m.vis.map.znear*ext, m.vis.map.zfar*ext
    depth = near / (1 - d*(1 - near/far))                      # 深度バッファ → 距離(m)
    return rgb[::-1], depth[::-1]                                # 画像の上下を通常の向きに
def to_world(depth):
    f = (RES/2) / np.tan(np.radians(TOPCAM_FOVY)/2); c = (RES-1)/2
    v, u = np.mgrid[0:RES, 0:RES]
    x = RING_C[0] + (u - c)*depth/f; y = RING_C[1] - (v - c)*depth/f; z = TOPCAM_Z - depth
    return x, y, z
def _hull(P):
    P = sorted(map(tuple, P)); cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])
def _square_fit(P):                                              # 最小外接長方形の向き(=正方形の向き、90°周期)と中心
    Hh = _hull(P); best = None
    for i in range(len(Hh)):
        e = Hh[(i+1) % len(Hh)] - Hh[i]; a = np.arctan2(e[1], e[0])
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]); Q = Hh @ R.T
        area = np.ptp(Q[:, 0]) * np.ptp(Q[:, 1])
        if best is None or area < best[0]:
            mid = np.array([Q[:, 0].min()+Q[:, 0].max(), Q[:, 1].min()+Q[:, 1].max()])/2
            best = (area, a, mid @ R)                             # 回転座標の中心を元の座標へ
    a = (best[1] + np.pi/4) % (np.pi/2) - np.pi/4
    return best[2], a
def observe(env):
    """戻り値: [{"xy","z_top","yaw","color","level","in_ring"}] 台上の石と塔の最上段"""
    rgb, depth = capture(env); x, y, z = to_world(depth)
    r = np.hypot(x - RING_C[0], y - RING_C[1])
    mask = (z > TABLE_Z + 0.01) & (z < TABLE_Z + 0.40) & ~((r > RING_R - RING_r - 0.004) & (r < RING_R + RING_r + 0.004)) & (r < 0.36)
    lab, n = ndimage.label(mask); out = []
    for k in range(1, n+1):
        sel = lab == k
        if sel.sum() < 80: continue
        zt = z[sel].max(); top = sel & (z > zt - 0.004)
        P = np.c_[x[top], y[top]]
        if len(P) < 30: continue
        c, yaw = _square_fit(P); lum = rgb[top].mean()
        out.append(dict(xy=c, z_top=float(zt), yaw=float(yaw), color="B" if lum < 110 else "W",
                        level=int(round((zt - TABLE_Z)/(2*H))), in_ring=bool(np.hypot(*(c - RING_C)) < RING_R - RING_r), npix=int(top.sum())))
    return out
def detect_ring(env, rgbd=None):
    """真上の深度画像から円環を見つけ、その中心と半径を世界座標で求める(円環の位置は事前知識として使わない)。
    高さが円環の頂部(テーブル+2r)付近の画素のうち最大の連結成分を円環とみなし、円を最小二乗で当てはめる"""
    rgb, depth = rgbd if rgbd is not None else capture(env); x, y, z = to_world(depth)
    band = (z > TABLE_Z + 2*RING_r - 0.012) & (z < TABLE_Z + 2*RING_r + 0.004)
    lab, n = ndimage.label(band)
    if n == 0: return None
    k = 1 + int(np.argmax(ndimage.sum(band, lab, range(1, n+1)))); sel = lab == k
    X, Y = x[sel], y[sel]; A = np.c_[2*X, 2*Y, np.ones_like(X)]
    cx, cy, c = np.linalg.lstsq(A, X**2 + Y**2, rcond=None)[0]
    return dict(center=np.array([cx, cy]), radius=float(np.sqrt(c + cx**2 + cy**2)), npix=int(sel.sum()))
