# 五段タワーのスクリプト制御: キューブを1個ずつ円環の外から中央へ運び、下の段に対して指定角だけ回して積む
import numpy as np
from twist_env import H, quat2mat, yaw_of, wrap90
from expert import R_DOWN, Rz, mat2aa
from tower_env import TABLE_Z, RING_C
from geo import geo_yaw                             # 形としての向き(転がって寝た石でも正しく測る)
class TowerExpert:
    def __init__(self, env, comp, kp=10.0, kr=2.0, max_phase=200):
        """comp: 角度の並び[0,18,..] または構成案[{"rel_deg","dx_mm","dy_mm",...}](1段目のrel_degは絶対角、dx/dyは下の段からのずらし)"""
        if not isinstance(comp[0], dict): comp = [dict(rel_deg=d, dx_mm=0, dy_mm=0) for d in comp]
        self.env, self.comp = env, comp
        self.rel = [np.radians(l["rel_deg"]) for l in comp]; self.off = [np.array([l["dx_mm"], l["dy_mm"]])/1000 for l in comp]
        self.kp, self.kr, self.maxp = kp, kr, max_phase
        self.k = 0; self.ph = 0; self.t = 0; self.yaw_t = None; self.hold = None; self.timeouts = []; self.done = False
    def _next(self, timeout=False):
        if timeout: self.timeouts.append((self.k, self.ph))
        self.ph += 1; self.t = 0; self.yaw_t = None; self.hold = None
        if self.ph > 7: self.k += 1; self.ph = 0
        if self.k >= len(self.env.cubes): self.done = True
    def act(self, o):
        if self.done: return np.r_[np.zeros(6), -1.0]
        k = self.k; P, qP = self.env.pose(k); yP = yaw_of(qP)
        e, R = o["robot0_eef_pos"], quat2mat(o["robot0_eef_quat"]); yE = np.arctan2(R[1, 0], R[0, 0])
        if k == 0: D = np.r_[RING_C + self.off[0], TABLE_Z + H]; ygoal = self.rel[0]
        else:
            Db, qb = self.env.pose(k-1); D = Db + np.r_[self.off[k], 2*H]; ygoal = yaw_of(qb) + self.rel[k]
        z_goal = D[2] + 0.003; safe = max(TABLE_Z + 0.20, z_goal + 0.13)     # 運搬高さ: 塔の上を越える
        g = -1.0; ph = self.ph
        if self.yaw_t is None:
            if ph == 0: self.yaw_t = yE + wrap90(yP - yE)
            elif ph >= 4: self.yaw_t = yE + wrap90(ygoal - yP)
            else: self.yaw_t = yE
        if   ph == 0: p = P + [0, 0, 0.08]
        elif ph == 1: p = P + [0, 0, 0.005]
        elif ph == 2: p = self.hold if self.hold is not None else e; g = 1.0
        elif ph == 3: p = np.r_[e[:2], safe]; g = 1.0
        elif ph == 4: p = np.r_[e[:2] + D[:2] - P[:2], safe]; g = 1.0
        elif ph == 5: p = e + np.r_[D[:2] - P[:2], z_goal - P[2]]; g = 1.0
        elif ph == 6: p = self.hold if self.hold is not None else e
        else:         p = np.r_[(self.hold if self.hold is not None else e)[:2], safe]
        if ph in (2, 6, 7) and self.hold is None: self.hold = e.copy()
        dp, dr = p - e, mat2aa(Rz(self.yaw_t) @ R_DOWN @ R.T)
        a = np.zeros(7); a[:3] = np.clip(dp*self.kp, -1, 1); a[3:6] = np.clip(dr*self.kr, -1, 1); a[6] = g
        pe, re = np.linalg.norm(dp), np.linalg.norm(dr); self.t += 1
        ok = {0: pe < 0.01 and re < 0.03, 1: pe < 0.006, 2: self.t >= 15, 3: pe < 0.015,
              4: pe < 0.008 and re < 0.02, 5: pe < 0.004, 6: self.t >= 15, 7: pe < 0.015}[ph]
        if ok: self._next()
        elif self.t >= self.maxp: self._next(timeout=True)
        return a
def tower_report(env, comp):                      # 各段の、目標(下の段+ずらし)に対する位置誤差・高さ・角度誤差
    if not isinstance(comp[0], dict): comp = [dict(rel_deg=d, dx_mm=0, dy_mm=0) for d in comp]
    rel_degs = [l["rel_deg"] for l in comp]; off = [np.array([l["dx_mm"], l["dy_mm"]])/1000 for l in comp]; rows = []
    for k in range(len(env.cubes)):
        P, q = env.pose(k)
        if k == 0:
            rows.append(dict(level=1, xy_mm=float(np.linalg.norm(P[:2]-RING_C-off[0])*1000), dz_mm=float((P[2]-TABLE_Z-H)*1000), yaw_err=float(np.degrees(wrap90(yaw_of(q)-np.radians(rel_degs[0]))))))
        else:
            Pb, qb = env.pose(k-1)
            rows.append(dict(level=k+1, xy_mm=float(np.linalg.norm(P[:2]-Pb[:2]-off[k])*1000), dz_mm=float((P[2]-Pb[2]-2*H)*1000),
                             yaw_err=float(np.degrees(wrap90(yaw_of(q)-yaw_of(qb)-np.radians(rel_degs[k]))))))
    ok = all(abs(r["dz_mm"]) < 5 and r["xy_mm"] < 10 and abs(r["yaw_err"]) < 5 for r in rows)
    return ok, rows
def tower_report_by_height(env, comp):            # 石の個体ではなく「積まれた高さの順」で評価(見て積む場合は色で石を選ぶため)
    if not isinstance(comp[0], dict): comp = [dict(rel_deg=d, dx_mm=0, dy_mm=0) for d in comp]
    poses = sorted([env.pose(i) for i in range(len(env.cubes))], key=lambda p: p[0][2]); rows = []
    for k, (P, q) in enumerate(poses):
        if k == 0:
            rows.append(dict(level=1, xy_mm=float(np.linalg.norm(P[:2]-RING_C)*1000), dz_mm=float((P[2]-TABLE_Z-H)*1000),
                             yaw_err=float(np.degrees(wrap90(geo_yaw(q)-np.radians(comp[0]["rel_deg"]))))))
        else:
            Pb, qb = poses[k-1]
            rows.append(dict(level=k+1, xy_mm=float(np.linalg.norm(P[:2]-Pb[:2])*1000), dz_mm=float((P[2]-Pb[2]-2*H)*1000),
                             yaw_err=float(np.degrees(wrap90(geo_yaw(q)-geo_yaw(qb)-np.radians(comp[k]["rel_deg"]))))))
    ok = all(abs(r["dz_mm"]) < 5 and r["xy_mm"] < 10 and abs(r["yaw_err"]) < 5 for r in rows)
    return ok, rows
