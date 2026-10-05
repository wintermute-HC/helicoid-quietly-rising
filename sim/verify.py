# 構成案の高速検査: (1)重心と支持面から「倒れにくさの余裕(mm)」を解析的に計算 (2)ロボットなしの物理シミュレーションで立つか確認
import numpy as np, mujoco
from tower_env import RING_C, RING_R, RING_r, NSEG, TABLE_Z, H
MAX_OFF_MM = 20.0
SOLREF, SOLIMP = "0.004 1", "0.99 0.99 0.001"     # 大理石を想定した硬い接触(既定の柔らかい接触では積み石がじわじわ傾き崩れる)
def targets(comp):
    """comp: [{"color","rel_deg","dx_mm","dy_mm"}]  1段目のrel_degは絶対角、dx/dyは下の段(1段目は円環中心)からのずらし"""
    out, yaw, xy = [], 0.0, RING_C.copy()
    for k, l in enumerate(comp):
        yaw = l["rel_deg"] if k == 0 else yaw + l["rel_deg"]
        xy = xy + np.array([l["dx_mm"], l["dy_mm"]]) / 1000.0
        out.append(dict(xy=xy.copy(), z=TABLE_Z + H + 2*H*k, yaw=yaw))
    return out
def _square(c, yaw_deg):
    t = np.radians(yaw_deg); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (np.array([[-H, -H], [H, -H], [H, H], [-H, H]]) @ R.T) + c
def _clip(P, Q):                                  # 凸多角形の共通部分(Sutherland-Hodgman、反時計回り)
    out = P
    for i in range(len(Q)):
        a, b = Q[i], Q[(i+1) % len(Q)]; inp, out = out, []
        side = lambda p: (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        for j in range(len(inp)):
            p, q = inp[j], inp[(j+1) % len(inp)]; sp, sq = side(p), side(q)
            if sp >= 0: out.append(p)
            if sp * sq < 0: out.append(p + (q-p) * sp / (sp - sq))
        if not out: return []
    return out
def _margin(pt, poly):                            # 点から多角形の辺までの符号付き距離(内側が正)
    if len(poly) < 3: return -1.0
    d = []
    for i in range(len(poly)):
        a, b = np.asarray(poly[i]), np.asarray(poly[(i+1) % len(poly)]); e = b - a
        d.append(((b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])) / (np.linalg.norm(e) + 1e-12))
    return float(min(d))
def analyze(comp):
    """各境界面で、それより上の段の重心が支持面(上下の正方形の重なり)の内側に何mm余裕を持つか"""
    T = targets(comp); rep = []
    for k in range(len(T) - 1):
        sup = _clip(list(_square(T[k]["xy"], T[k]["yaw"])), list(_square(T[k+1]["xy"], T[k+1]["yaw"])))
        com = np.mean([t["xy"] for t in T[k+1:]], axis=0)
        rep.append(dict(interface=f"{k+1}-{k+2}段", margin_mm=round(_margin(com, sup)*1000, 1)))
    inner = RING_R - RING_r
    ring_ok = all(np.max(np.linalg.norm(_square(T[k]["xy"], T[k]["yaw"]) - RING_C, axis=1)) < inner - 0.005 for k in range(min(2, len(T))))
    offs_ok = all(abs(l["dx_mm"]) <= MAX_OFF_MM and abs(l["dy_mm"]) <= MAX_OFF_MM for l in comp)
    return dict(min_margin_mm=min(r["margin_mm"] for r in rep), interfaces=rep, ring_clear=ring_ok, offsets_in_range=offs_ok)
def physics(comp, seconds=10.0):
    """ロボットなしでキューブを目標位置に置き、物理だけで立つかを確かめる"""
    T = targets(comp); g = ""
    for i in range(NSEG):
        t0, t1 = 2*np.pi*i/NSEG, 2*np.pi*(i+1)/NSEG
        g += f'<geom type="capsule" size="{RING_r}" fromto="{RING_R*np.cos(t0)} {RING_R*np.sin(t0)} 0 {RING_R*np.cos(t1)} {RING_R*np.sin(t1)} 0"/>'
    bodies = "".join(f'<body pos="{t["xy"][0]} {t["xy"][1]} {t["z"] + 0.0005*(k+1)}" euler="0 0 {t["yaw"]}"><freejoint/>'
                     f'<geom type="box" size="{H} {H} {H}" density="2700" friction="1 0.005 0.0001" solref="{SOLREF}" solimp="{SOLIMP}"/></body>' for k, t in enumerate(T))
    xml = f'''<mujoco><compiler angle="degree"/><option timestep="0.002"/><worldbody>
<geom type="plane" size="1 1 0.01" pos="0 0 {TABLE_Z}" friction="1 0.005 0.0001" solref="{SOLREF}" solimp="{SOLIMP}"/>
<body pos="{RING_C[0]} {RING_C[1]} {TABLE_Z + RING_r}">{g}</body>{bodies}</worldbody></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    for _ in range(int(seconds / m.opt.timestep)): mujoco.mj_step(m, d)
    res = []
    for k, t in enumerate(T):
        p = d.xpos[k+2]; q = d.xquat[k+2]; tilt = np.degrees(2*np.arccos(min(1, np.sqrt(q[0]**2 + q[3]**2))))
        res.append(dict(level=k+1, drift_mm=round(float(np.linalg.norm(p[:2]-t["xy"])*1000), 1),
                        drop_mm=round(float((t["z"]-p[2])*1000), 1), tilt_deg=round(float(tilt), 1)))
    stands = all(r["drift_mm"] < 5 and r["drop_mm"] < 5 and r["tilt_deg"] < 3 for r in res)
    return dict(stands=stands, levels=res)
def check(comp, need_margin_mm=8.0):
    a = analyze(comp); p = physics(comp)
    ok = a["ring_clear"] and a["offsets_in_range"] and p["stands"] and a["min_margin_mm"] >= need_margin_mm
    return dict(ok=ok, need_margin_mm=need_margin_mm, analysis=a, physics=p)
