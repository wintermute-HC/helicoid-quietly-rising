# 第1段の学習: 共通部。方策が見る世界は「円環から見た位置」だけ(円環座標)。
# 上位(知覚・どの石をどこへ何度で置くか・置いた後の検査)は見て積む制御のまま。
# 学習する方策は、掴む・運ぶ・回す・置く・離れる、の運動そのもの。
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import numpy as np
from twist_env import quat2mat, wrap90
from tower_env import TABLE_Z
F_DIM = 29
def eef(o):
    e, R = o["robot0_eef_pos"], quat2mat(o["robot0_eef_quat"]); return e, np.arctan2(R[1, 0], R[0, 0])
def gap(o): q = o["robot0_gripper_qpos"]; return float(q[0] - q[1])
class Tracker:
    """運ぶ石の推定位置 S を保つ。掴む前は知覚した値、掴んでいる間は手先に固定、離した後はその場に固定。
    「掴んでいる」は、閉じる指令を出していて、指の開きが石の幅で止まっていることから判断する(シミュレータの正解は使わない)"""
    def __init__(self, P0, yP0):
        self.S, self.yS = np.array(P0, float), float(yP0); self.off = None; self.held_once = False; self.lifted = False; self.prev_g = -1.0
    def update(self, o):
        e, yE = eef(o); held = self.prev_g > 0 and 0.035 < gap(o) < 0.065
        if held:
            if self.off is None: self.off, self.dy = self.S - e, self.yS - yE; self.held_once = True
            self.S, self.yS = e + self.off, yE + self.dy
        elif self.off is not None: self.off = None                       # 離した: S はその場に止まる
        return held
def extra(o, tr, D):
    """v7追加: 先生が使っているのに生徒に見えていなかった情報(傾き)と、mm単位の距離を拡大した入力"""
    from expert import R_DOWN, Rz, mat2aa
    e, yE = eef(o); R = quat2mat(o["robot0_eef_quat"])
    tilt = mat2aa(Rz(yE) @ R_DOWN @ R.T)
    dg = tr.S + np.array([0, 0, 0.005]) - e
    dp = np.r_[D[:2] - tr.S[:2], D[2] + 0.003 - tr.S[2]]
    return [*tilt, *np.tanh(dg/0.01), *np.tanh(dp/0.01), np.tanh(np.linalg.norm(dg)/0.005), np.tanh(np.linalg.norm(dp)/0.005)]
def features(o, tr, axis, D, ygoal):
    e, yE = eef(o); a3 = np.r_[axis, TABLE_Z]
    safe = max(TABLE_Z + 0.20, D[2] + 0.133)
    if tr.off is not None and safe - e[2] < 0.02: tr.lifted = True       # 掴んだ石を安全高さまで持ち上げた(記憶)
    return np.r_[e - a3, gap(o), tr.prev_g, tr.S - e, D - tr.S, wrap90(tr.yS - yE), wrap90(ygoal - tr.yS), safe - e[2],
                 np.sin(4*(ygoal - tr.yS)), np.cos(4*(ygoal - tr.yS)), float(tr.held_once), float(tr.lifted), np.asarray(extra(o, tr, D))].astype(np.float32)
def act_done(o, tr, D):
    e, _ = eef(o); safe = max(TABLE_Z + 0.20, D[2] + 0.133)
    return tr.held_once and tr.off is None and tr.prev_g < 0 and e[2] > safe - 0.02
class NumpyMLP:
    """学習済みの重み(npz)で推論する。シミュレータ側の環境には torch を入れない"""
    def __init__(self, path):
        z = np.load(path); self.W = [z[f"W{i}"] for i in range(z["n"])]; self.b = [z[f"b{i}"] for i in range(z["n"])]
        self.mu, self.sd = z["mu"], z["sd"]
    def __call__(self, f):
        h = (f - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1: h = np.maximum(h, 0)
        a = np.clip(h, -1, 1); a[6] = 1.0 if a[6] > 0 else -1.0; return a
