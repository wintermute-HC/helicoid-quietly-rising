import numpy as np
from twist_env import quat2mat, wrap90
def geo_yaw(qxyzw):                    # 立方体の形としての向き: 最も鉛直に近い軸を除いた水平の軸の方位(転がって寝た石にも正しい)
    R = quat2mat(qxyzw); v = int(np.argmax(np.abs(R[2, :]))); h = R[:, (v+1) % 3]
    return np.arctan2(h[1], h[0])
