# 構成案の完成予想図を描く(ロボットなし・物理なし、数秒): 作品カメラ(斜め上)と低い横からの2視点
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, io, base64, numpy as np, mujoco
from PIL import Image
from tower_env import RING_C, RING_R, RING_r, NSEG, TABLE_Z, H, COLORS, textures, lookat_quat, torus_mesh
from verify import targets
LOW_POS = np.array([RING_C[0] - 0.05, -0.62, 0.98])     # 低い横からの視点: ねじれの輪郭が見える
ART_POS = np.array([0.62, -0.58, 1.22]); ART_TGT = np.array([RING_C[0], RING_C[1], TABLE_Z + 0.11])
def _xml(levels):
    tx = textures(); T = targets(levels)
    ring = '<geom type="mesh" mesh="torus" material="W2"/>'
    boxes = "".join(f'<body pos="{t["xy"][0]} {t["xy"][1]} {t["z"]}" euler="0 0 {t["yaw"]}"><geom type="box" size="{H} {H} {H}" material="{c}"/></body>'
                    for t, c in zip(T, COLORS))
    q = lookat_quat(ART_POS, ART_TGT); ql = lookat_quat(LOW_POS, ART_TGT + [0, 0, 0.02])
    return f'''<mujoco><compiler angle="degree"/><visual><global offwidth="640" offheight="640"/><quality shadowsize="4096"/>
<headlight ambient="0.3 0.3 0.3" diffuse="0.2 0.2 0.2" specular="0.05 0.05 0.05"/></visual>
<asset><texture type="skybox" builtin="gradient" rgb1="0.92 0.92 0.92" rgb2="0.72 0.72 0.74" width="256" height="256"/>
<texture name="tw" type="cube" file="{tx['W']}"/><texture name="tb" type="cube" file="{tx['B']}"/>
<material name="W" texture="tw" specular="0.35" shininess="0.5"/><material name="B" texture="tb" specular="0.4" shininess="0.6"/>
<texture name="tw2" type="2d" file="{tx['W']}"/><material name="W2" texture="tw2" specular="0.35" shininess="0.5"/><mesh name="torus" file="{torus_mesh()}"/>
<texture name="fl" type="2d" builtin="flat" rgb1="0.86 0.85 0.83" width="16" height="16"/><material name="F" texture="fl" specular="0.05"/></asset>
<worldbody><light pos="0.9 -0.6 1.6" dir="-0.9 0.6 -0.8" diffuse="1 0.97 0.92" castshadow="true"/>
<geom type="plane" size="1 1 0.01" pos="0 0 {TABLE_Z}" material="F"/>
<body pos="{RING_C[0]} {RING_C[1]} {TABLE_Z + RING_r}">{ring}</body>{boxes}
<camera name="art" pos="{ART_POS[0]} {ART_POS[1]} {ART_POS[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}" fovy="40"/>
<camera name="low" pos="{LOW_POS[0]} {LOW_POS[1]} {LOW_POS[2]}" quat="{ql[0]} {ql[1]} {ql[2]} {ql[3]}" fovy="40"/></worldbody></mujoco>'''
def render(levels, size=384):
    m = mujoco.MjModel.from_xml_string(_xml(levels)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, size, size); ims = []
    for cam in ("art", "low"): r.update_scene(d, camera=cam); ims.append(r.render().copy())
    del r; return np.concatenate(ims, 1)
def png_b64(img):
    b = io.BytesIO(); Image.fromarray(img).save(b, format="PNG"); return base64.b64encode(b.getvalue()).decode()
