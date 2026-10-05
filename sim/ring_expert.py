# 複数のピック&プレースを順に実行するスクリプト制御(円環を越える運搬高さを持つ)
import numpy as np
from twist_env import H, quat2mat, yaw_of, wrap90
from expert import R_DOWN, Rz, mat2aa
TABLE_Z = 0.8
SAFE_Z = TABLE_Z + 0.20          # 運搬時のグリッパ高さ(キューブ底≒テーブル+0.175 > 円環頂部+0.05)
class RingExpert:
    """tasks: [(物体名, 置き先xy or 別物体名, 置いた後の目標ヨー(rad, world) or ('rel', 物体名, rad))]"""
    def __init__(self, tasks, kp=10.0, kr=2.0, max_phase=150):
        self.tasks=tasks; self.ti=0; self.kp=kp; self.kr=kr; self.maxp=max_phase
        self.ph=0; self.t=0; self.yaw_t=None; self.hold=None; self.timeouts=[]; self.done=False
    def _next(self, timeout=False):
        if timeout: self.timeouts.append((self.ti, self.ph))
        self.ph+=1; self.t=0; self.yaw_t=None; self.hold=None
        if self.ph > 7: self.ti += 1; self.ph = 0
        if self.ti >= len(self.tasks): self.done = True
    def act(self, o):
        if self.done: return np.r_[np.zeros(6), -1.0]
        obj, dst, yspec = self.tasks[self.ti]
        e, R = o["robot0_eef_pos"], quat2mat(o["robot0_eef_quat"])
        P, yP = o[f"{obj}_pos"], yaw_of(o[f"{obj}_quat"]); yE = np.arctan2(R[1,0], R[0,0])
        if isinstance(dst, str): D = o[f"{dst}_pos"]; dz_goal = D[2] + 2*H + 0.003     # 別物体の上
        else: D = np.r_[dst, TABLE_Z + H]; dz_goal = TABLE_Z + H + 0.003                # テーブル上
        ygoal = yspec if not isinstance(yspec, tuple) else yaw_of(o[f"{yspec[1]}_quat"]) + yspec[2]
        g = -1.0; ph = self.ph
        if self.yaw_t is None:
            if ph == 0: self.yaw_t = yE + wrap90(yP - yE)
            elif ph >= 4: self.yaw_t = yE + wrap90(ygoal - yP)
            else: self.yaw_t = yE
        if   ph == 0: p = P + [0, 0, 0.08]
        elif ph == 1: p = P + [0, 0, 0.005]
        elif ph == 2: p = self.hold if self.hold is not None else e; g = 1.0
        elif ph == 3: p = np.r_[e[:2], SAFE_Z]; g = 1.0                               # 真上へ持ち上げ
        elif ph == 4: p = np.r_[e[:2] + D[:2] - P[:2], SAFE_Z]; g = 1.0              # 高さを保って水平移動
        elif ph == 5: p = e + np.r_[D[:2] - P[:2], dz_goal - P[2]]; g = 1.0          # 降ろす
        elif ph == 6: p = self.hold if self.hold is not None else e
        elif ph == 7: p = np.r_[(self.hold if self.hold is not None else e)[:2], SAFE_Z]
        if ph in (2, 6) and self.hold is None: self.hold = e.copy()
        if ph == 7 and self.hold is None: self.hold = e.copy()
        dp, dr = p - e, mat2aa(Rz(self.yaw_t) @ R_DOWN @ R.T)
        a = np.zeros(7); a[:3] = np.clip(dp*self.kp, -1, 1); a[3:6] = np.clip(dr*self.kr, -1, 1); a[6] = g
        pe, re = np.linalg.norm(dp), np.linalg.norm(dr); self.t += 1
        ok = {0: pe<0.01 and re<0.03, 1: pe<0.006, 2: self.t>=15, 3: pe<0.015,
              4: pe<0.01 and re<0.02, 5: pe<0.004, 6: self.t>=15, 7: pe<0.015}.get(ph, False)
        if ok: self._next()
        elif self.t >= self.maxp: self._next(timeout=True)
        return a
