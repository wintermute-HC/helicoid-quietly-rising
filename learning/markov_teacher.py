# 状態だけで決まる先生(マルコフ化): 生徒が見る情報(円環座標の特徴量と同じもの)だけから行動を決める。
# 局面番号もタイマーも持たないので、生徒が迷い込んだどの状態でも正しい答え(添削)を返せる。
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import numpy as np
from expert import R_DOWN, Rz, mat2aa
from twist_env import quat2mat, wrap90
from tower_env import TABLE_Z
from learn_common import eef, gap
KP, KR = 10.0, 2.0
def teacher(o, tr, D, ygoal):
    e, yE = eef(o); R = quat2mat(o["robot0_eef_quat"]); S = tr.S; held = tr.off is not None
    z_goal = D[2] + 0.003; safe = max(TABLE_Z + 0.20, z_goal + 0.13); g = -1.0; yaw_t = yE
    if not held and not tr.held_once:                                   # 掴むまで
        dxy = np.linalg.norm(S[:2] - e[:2]); yerr = wrap90(tr.yS - yE); yaw_t = yE + yerr
        grasp = S + np.array([0, 0, 0.005])
        if tr.prev_g > 0 and gap(o) >= 0.065: p, g = grasp, 1.0              # 閉じている途中
        elif tr.prev_g > 0: p = np.r_[e[:2], S[2] + 0.08]                     # 閉じきったのに掴めていない: 開いて上がる
        elif dxy > 0.012 and e[2] < S[2] + 0.06: p = np.r_[e[:2], S[2] + 0.08]   # 低い位置で横にずれている: まず真上へ
        elif dxy > 0.01 or (abs(yerr) > 0.03 and e[2] > S[2] + 0.07): p = S + np.array([0, 0, 0.08])
        else:
            p = grasp
            if np.linalg.norm(grasp - e) < 0.006: g = 1.0                   # 掴む位置に着いた: 閉じる
    elif held:                                                         # 掴んでいる
        g = 1.0; yerr = wrap90(ygoal - tr.yS); dxy = D[:2] - S[:2]
        if not tr.lifted: p = np.r_[e[:2], safe]                           # まず安全な高さへ
        else:
            yaw_t = yE + yerr; low = e[2] < safe - 0.03
            if low and np.linalg.norm(dxy) > 0.015: p = np.r_[e[:2], safe]   # 低い位置で大きくずれた: 上がり直す
            elif (not low) and (np.linalg.norm(dxy) > 0.008 or abs(yerr) > 0.02): p = np.r_[e[:2] + dxy, safe]   # 上空で運び、回す
            else:
                d = np.r_[dxy, z_goal - S[2]]; p = e + d                     # 降ろしながら合わせる
                if np.linalg.norm(d) < 0.004: g = -1.0                      # 置く位置に着いた: 離す
    else:                                                              # 離した後
        p = e.copy() if gap(o) < 0.07 else np.r_[e[:2], safe]               # 指が開ききるまで待ち、それから上がる
    dp, dr = p - e, mat2aa(Rz(yaw_t) @ R_DOWN @ R.T)
    a = np.zeros(7); a[:3] = np.clip(dp*KP, -1, 1); a[3:6] = np.clip(dr*KR, -1, 1); a[6] = g
    return a
