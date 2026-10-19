# 漏斗型の先生: 状態だけで決まり、しかも急な切り替えがない(高さを横ずれに応じて連続的に変える)。
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import numpy as np
from expert import R_DOWN, Rz, mat2aa
from twist_env import quat2mat, wrap90
from tower_env import TABLE_Z
from learn_common import eef, gap
KP, KR = 10.0, 2.0
c01 = lambda x: float(np.clip(x, 0.0, 1.0))
def teacher(o, tr, D, ygoal):
    e, yE = eef(o); R = quat2mat(o["robot0_eef_quat"]); S = tr.S; held = tr.off is not None
    z_goal = D[2] + 0.003; safe = max(TABLE_Z + 0.20, z_goal + 0.13); g = -1.0; yaw_t = yE
    if not held and not tr.held_once:                                   # 掴むまで: ずれが大きいほど高く(8cm)、揃うにつれて降りる
        dxy = np.linalg.norm(S[:2] - e[:2]); yerr = wrap90(tr.yS - yE); yaw_t = yE + yerr
        grasp = S + np.array([0, 0, 0.005])
        if tr.prev_g > 0 and gap(o) >= 0.065: p, g = grasp, 1.0              # 閉じている途中
        elif tr.prev_g > 0: p = np.r_[e[:2], S[2] + 0.08]                     # 掴み損ね: 開いて上がる
        else:
            al = max(c01((dxy - 0.003)/0.012), c01((abs(yerr) - 0.02)/0.08))
            p = np.r_[S[:2], S[2] + 0.005 + 0.075*al]
            if np.linalg.norm(grasp - e) < 0.006: g = 1.0                   # 掴む位置に着いた: 閉じる
    elif held:
        g = 1.0; yerr = wrap90(ygoal - tr.yS); dv = D[:2] - S[:2]; dxy = np.linalg.norm(dv)
        if not tr.lifted: p = np.r_[e[:2], safe]
        else:                                                          # 置くまで: ずれが大きいほど石を高く(13cm)、揃うにつれて降ろす
            yaw_t = yE + yerr
            al = max(c01((dxy - 0.002)/0.010), c01((abs(yerr) - 0.01)/0.04))
            zs = z_goal + 0.13*al; p = e + np.r_[dv, zs - S[2]]
            if np.linalg.norm(np.r_[dv, z_goal - S[2]]) < 0.004: g = -1.0
    else:
        p = e.copy() if gap(o) < 0.07 else np.r_[e[:2], safe]
    dp, dr = p - e, mat2aa(Rz(yaw_t) @ R_DOWN @ R.T)
    a = np.zeros(7); a[:3] = np.clip(dp*KP, -1, 1); a[3:6] = np.clip(dr*KR, -1, 1); a[6] = g
    return a
