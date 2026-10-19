import numpy as np
from twist_env import H, quat2mat, yaw_of, wrap90
R_DOWN = np.array([[1,0,0],[0,-1,0],[0,0,-1]], float)   # グリッパ真下向き・ヨー0
def Rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def mat2aa(R):
    ang = np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
    if ang < 1e-6: return np.zeros(3)
    return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(ang))*ang
class Expert:
    """下の立方体Bに対し、上の立方体AをBから rel_deg 度ずらして置く"""
    def __init__(self, rel_deg, kp=10.0, kr=2.0, max_phase=150):
        self.rel=np.radians(rel_deg); self.kp=kp; self.kr=kr; self.maxp=max_phase
        self.ph=0; self.t=0; self.yaw_t=None; self.hold=None; self.timeouts=[]; self.done=False
    def _next(self, timeout=False):
        if timeout: self.timeouts.append(self.ph)
        self.ph+=1; self.t=0; self.yaw_t=None; self.hold=None
    def act(self, o):
        e, R = o["robot0_eef_pos"], quat2mat(o["robot0_eef_quat"])
        A, B = o["cubeA_pos"], o["cubeB_pos"]
        yA, yB, yE = yaw_of(o["cubeA_quat"]), yaw_of(o["cubeB_quat"]), np.arctan2(R[1,0],R[0,0])
        g = -1.0; ph = self.ph
        if self.yaw_t is None:
            if ph == 0: self.yaw_t = yE + wrap90(yA - yE)                 # 把持: Aの面に合わせる
            elif ph >= 4: self.yaw_t = yE + wrap90(yB + self.rel - yA)    # 設置: Aを目標角へ
            else: self.yaw_t = yE
        Az_goal = B[2] + 2*H + 0.003
        if   ph == 0: p = A + [0,0,0.08]
        elif ph == 1: p = A + [0,0,0.005]
        elif ph == 2: p = self.hold if self.hold is not None else e; g = 1.0
        elif ph == 3: p = np.r_[e[:2], B[2]+2*H+0.10+(e[2]-A[2])]; g = 1.0
        elif ph == 4: p = e + np.r_[B[:2]-A[:2], Az_goal+0.06-A[2]]; g = 1.0
        elif ph == 5: p = e + np.r_[B[:2]-A[:2], Az_goal-A[2]]; g = 1.0
        elif ph == 6: p = self.hold if self.hold is not None else e
        elif ph == 7: p = (self.hold if self.hold is not None else e) + [0,0,0.10]
        else: self.done = True; p = e
        if ph in (2,6,7) and self.hold is None: self.hold = e.copy()
        dp, dr = p - e, mat2aa(Rz(self.yaw_t) @ R_DOWN @ R.T)
        a = np.zeros(7); a[:3]=np.clip(dp*self.kp,-1,1); a[3:6]=np.clip(dr*self.kr,-1,1); a[6]=g
        pe, re = np.linalg.norm(dp), np.linalg.norm(dr); self.t += 1
        ok = {0: pe<0.01 and re<0.03, 1: pe<0.006, 2: self.t>=15, 3: pe<0.015,
              4: pe<0.008 and re<0.02, 5: pe<0.004, 6: self.t>=15, 7: self.t>=25}.get(ph, False)
        if ok: self._next()
        elif self.t >= self.maxp and ph < 8: self._next(timeout=True)
        return a
def evaluate(o, rel_deg):
    A, B = o["cubeA_pos"], o["cubeB_pos"]
    yaw_err = np.degrees(wrap90(yaw_of(o["cubeA_quat"]) - yaw_of(o["cubeB_quat"]) - np.radians(rel_deg)))
    xy = np.linalg.norm(A[:2]-B[:2]); dz = A[2]-(B[2]+2*H)
    return dict(yaw_err=yaw_err, xy_mm=xy*1000, dz_mm=dz*1000, success=bool(abs(dz)<0.005 and xy<0.01 and abs(yaw_err)<5))
