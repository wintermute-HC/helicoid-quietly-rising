# 見て積む制御: 各石を運ぶ前に腕を視界の外へ退け、真上の深度カメラで石と塔を推定してから動く(シミュレータの正解は使わない)
import numpy as np
from twist_env import H, quat2mat, wrap90
from expert import R_DOWN, Rz, mat2aa
from tower_env import TABLE_Z, RING_C, COLORS
from perception import observe, detect_ring, capture
from twist_env import wrap90 as _w90
TOL_XY, TOL_YAW, MAX_FIX = 0.0025, np.radians(2.0), 2      # 置き直しの許容値(軸からの位置2.5mm・角度2°)と、1段あたりの置き直し回数の上限
VIEW_POS = np.array([-0.42, 0.0, TABLE_Z + 0.50])      # 観測時に腕を置く位置(ロボットの根元側の上空)
# 置くたびに生じるずれは、手で合わせた定数を使わず、作動の中で測って学ぶ(self.bias: 狙った位置と置かれた位置の差の平均)
NEUTRAL_POS = np.array([RING_C[0], RING_C[1], TABLE_Z + 0.30])   # 観測後に一度戻る中立位置(腕の姿勢を整える)
class SeeingTowerExpert:
    def __init__(self, env, comp, kp=10.0, kr=2.0, max_phase=200):
        if not isinstance(comp[0], dict): comp = [dict(rel_deg=d, dx_mm=0, dy_mm=0) for d in comp]
        self.env, self.rel = env, [np.radians(l["rel_deg"]) for l in comp]
        self.off = [np.array([l["dx_mm"], l["dy_mm"]])/1000 for l in comp]
        self.kp, self.kr, self.maxp = kp, kr, max_phase
        self.k = 0; self.stage = "view"; self.ph = 0; self.t = 0; self.yaw_t = None; self.hold = None
        self.timeouts, self.log, self.done = [], [], False; self.rings, self.fixes, self.miss = [], {}, {}; self.errs, self.aim = [], None
    def _bias(self): return np.mean(self.errs, axis=0) if self.errs else np.zeros(2)
    def _look(self):
        """円環を基準にする: 円環の中心から垂直な軸を立て、すべての石をその軸に対して置く(先に置いた石には頼らない)。
        置いた直後の観測で、最上段が軸から許容値以上ずれていれば、その石をつかみ直して置き直す"""
        rgbd = capture(self.env); obs = observe(self.env)
        ring = detect_ring(self.env, rgbd)
        if ring is not None: self.rings.append(ring["center"])
        self.axis = np.mean(self.rings, axis=0)                              # 観測のたびに円環の中心を測り、平均して軸を定める
        cum = np.cumsum(self.rel)                                             # 各段の絶対的な向き(世界座標)
        tower = sorted([o for o in obs if o["in_ring"] and np.linalg.norm(o["xy"] - self.axis) < 0.02], key=lambda o: o["z_top"])   # 軸上にあるものだけが塔
        n_lv = tower[-1]["level"] if tower else 0
        if n_lv < self.k and self.miss.get(self.k, 0) < 3:                             # 運んだはずの石が塔に無い(掴み損ね・落下): その段をもう一度
            self.miss[self.k] = self.miss.get(self.k, 0) + 1; self.log.append(dict(missing_level=self.k, tower_levels=n_lv)); self.k = n_lv
        if tower and self.k > 0 and n_lv == self.k:                           # 直前に置いた石を検査
            top = tower[-1]; lv = top["level"] - 1                                # 塔は一塊に見えるので、段は高さから求める
            if self.aim is not None: self.errs.append(top["xy"] - self.aim)         # 狙いと結果の差を記憶する(自己修正の内部化)
            exy = float(np.linalg.norm(top["xy"] - self.axis)); eyaw = float(_w90(top["yaw"] - cum[lv]))
            self.log.append(dict(check_level=lv+1, axis_err_mm=round(exy*1000, 2), yaw_err_deg=round(float(np.degrees(eyaw)), 2)))
            if (exy > TOL_XY or abs(eyaw) > TOL_YAW) and self.fixes.get(lv, 0) < MAX_FIX:
                self.fixes[lv] = self.fixes.get(lv, 0) + 1; self.k -= 1          # 同じ段をやり直す(置き終えると k が戻る)
                self.P0 = np.r_[top["xy"], top["z_top"] - H]; self.yP0 = top["yaw"]
                self.D = np.r_[self.axis - self._bias(), top["z_top"] - H]; self.ygoal = cum[lv]
                self.aim = self.D[:2].copy(); self.log.append(dict(refix_level=lv+1, bias_mm=(self._bias()*1000).round(2).tolist())); return
        if self.k >= len(COLORS): self.done = True; return                   # 全段が許容内に収まった
        need = COLORS[self.k]
        free = [o for o in obs if o["level"] == 1 and o["color"] == need and np.linalg.norm(o["xy"] - self.axis) >= 0.02]   # 円環の外、または円環内に落ちた石
        if not free:                                                          # 必要な色の石が見当たらない: 観測をやり直す(腕の影など)
            self.k = n_lv; self.log.append(dict(no_free_stone=need)); self.stage_retry = getattr(self, "stage_retry", 0) + 1
            if self.stage_retry > 3: self.done = True
            self.relook = True; return
        pick = min(free, key=lambda o: np.hypot(*(o["xy"] - self.axis)))
        self.P0 = np.r_[pick["xy"], pick["z_top"] - H]; self.yP0 = pick["yaw"]
        z_base = tower[-1]["z_top"] if tower else TABLE_Z
        self.D = np.r_[self.axis + self.off[self.k] - self._bias(), z_base + H]; self.ygoal = cum[self.k]
        self.aim = self.D[:2].copy()
        self.log.append(dict(k=self.k, bias_mm=(self._bias()*1000).round(2).tolist(), pick=pick["xy"].round(4).tolist(), pick_color=pick["color"], seen=len(obs), axis_mm=(self.axis*1000).round(2).tolist()))
    def act(self, o):
        if self.done: return np.r_[np.zeros(6), -1.0]
        e, R = o["robot0_eef_pos"], quat2mat(o["robot0_eef_quat"]); yE = np.arctan2(R[1, 0], R[0, 0])
        if self.stage == "view":                                           # 腕を退けて観測
            dp, dr = VIEW_POS - e, mat2aa(R_DOWN @ R.T); self.t += 1
            if (np.linalg.norm(dp) < 0.02 and self.t > 10) or self.t > 150:
                self._look(); self.stage = "neutral"; self.t = 0
                if self.done: return np.r_[np.zeros(6), -1.0]
                if getattr(self, "relook", False): self.relook = False; self.stage = "view"; self.t = 0
            a = np.zeros(7); a[:3] = np.clip(dp*self.kp, -1, 1); a[3:6] = np.clip(dr*self.kr, -1, 1); a[6] = -1; return a
        if self.stage == "neutral":                                        # 円環中央の上空へ戻り、腕を自然な姿勢に戻してから掴みに行く
            dp, dr = NEUTRAL_POS - e, mat2aa(R_DOWN @ R.T); self.t += 1
            if np.linalg.norm(dp) < 0.03 or self.t > 150: self.stage = "act"; self.ph = 0; self.t = 0
            a = np.zeros(7); a[:3] = np.clip(dp*self.kp, -1, 1); a[3:6] = np.clip(dr*self.kr, -1, 1); a[6] = -1; return a
        ph = self.ph
        if ph <= 2: P, yP = self.P0, self.yP0                              # 掴むまでは観測した値
        else: P, yP = e + self.grip_off, yE + self.dyaw                    # 掴んだ後は手先からの相対位置で追跡
        z_goal = self.D[2] + 0.003; safe = max(TABLE_Z + 0.20, z_goal + 0.13); g = -1.0
        if self.yaw_t is None:
            if ph == 0: self.yaw_t = yE + wrap90(yP - yE)
            elif ph >= 4: self.yaw_t = yE + wrap90(self.ygoal - yP)
            else: self.yaw_t = yE
        if   ph == 0: p = P + [0, 0, 0.08]
        elif ph == 1: p = P + [0, 0, 0.005]
        elif ph == 2: p = self.hold if self.hold is not None else e; g = 1.0
        elif ph == 3: p = np.r_[e[:2], safe]; g = 1.0
        elif ph == 4: p = np.r_[e[:2] + self.D[:2] - P[:2], safe]; g = 1.0
        elif ph == 5: p = e + np.r_[self.D[:2] - P[:2], z_goal - P[2]]; g = 1.0
        elif ph == 6: p = self.hold if self.hold is not None else e
        else:         p = np.r_[(self.hold if self.hold is not None else e)[:2], safe]
        if ph in (2, 6, 7) and self.hold is None: self.hold = e.copy()
        dp, dr = p - e, mat2aa(Rz(self.yaw_t) @ R_DOWN @ R.T)
        a = np.zeros(7); a[:3] = np.clip(dp*self.kp, -1, 1); a[3:6] = np.clip(dr*self.kr, -1, 1); a[6] = g
        pe, re = np.linalg.norm(dp), np.linalg.norm(dr); self.t += 1
        ok = {0: pe < 0.01 and re < 0.03, 1: pe < 0.006, 2: self.t >= 15, 3: pe < 0.015,
              4: pe < 0.008 and re < 0.02, 5: pe < 0.004, 6: self.t >= 15, 7: pe < 0.015}[ph]
        if ph == 2 and ok: self.grip_off = self.P0 - e; self.dyaw = self.yP0 - yE     # 掴んだ瞬間の相対関係を記憶
        if ok or self.t >= self.maxp:
            if not ok: self.timeouts.append((self.k, ph))
            self.ph += 1; self.t = 0; self.yaw_t = None; self.hold = None
            if self.ph > 7:
                self.k += 1; self.stage = "view"                      # 最後の石の後も一度見て、検査してから終える
        return a
