# 五段タワー環境: 白大理石の円環(衝突体)の外に黒白大理石キューブ5個、中央に螺旋状に積む
import os, numpy as np, xml.etree.ElementTree as ET
from PIL import Image
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
H = 0.025
TABLE_Z = 0.8
RING_C = np.array([-0.05, 0.0]); RING_R, RING_r, NSEG = 0.16, 0.025, 48
SPOT_R = 0.27
TOPCAM_Z, TOPCAM_FOVY = TABLE_Z + 1.30, 34.0
SPOT_DEG = [140, 100, 180, -100, -140]          # 1段目〜5段目のキューブの初期位置(円環中心から見た方位角)
COLORS = ["B", "W", "B", "W", "B"]              # 下から黒・白・黒・白・黒
TEX_DIR = "/tmp/pai_tex"
def _noise(rng, n, octv=6):
    out = np.zeros((n, n))
    for o in range(octv):
        s = 2**(o+2); g = rng.random((s, s))
        out += np.asarray(Image.fromarray((g*255).astype(np.uint8)).resize((n, n), Image.BICUBIC), float)/255/(2**o)
    return out/out.max()
def _marble(rng, n, base, veinc, freq=3, turb=4):
    y, x = np.mgrid[0:n, 0:n]/n; t = _noise(rng, n)
    s1 = np.abs(np.sin((x*0.8+y*0.35)*freq*np.pi + turb*t)); s2 = np.abs(np.sin((x*0.2-y*0.9)*freq*1.7*np.pi + turb*1.3*_noise(rng, n)))
    vein = np.clip(np.exp(-s1/0.05)*0.9 + np.exp(-s2/0.025)*0.45, 0, 1); cloud = (_noise(rng, n, 5)-0.5)*0.06
    return (np.clip(base*(1-vein[..., None]) + veinc*vein[..., None] + cloud[..., None], 0, 1)*255).astype(np.uint8)
def textures():                                   # 大理石テクスチャを決定的に生成(どの環境でも同じ柄)
    os.makedirs(TEX_DIR, exist_ok=True)
    w, b = f"{TEX_DIR}/white_marble.png", f"{TEX_DIR}/black_marble.png"
    if not (os.path.exists(w) and os.path.exists(b)):
        rng = np.random.default_rng(3)
        Image.fromarray(_marble(rng, 512, np.array([.93, .92, .90]), np.array([.45, .46, .50]))).save(w)
        Image.fromarray(_marble(rng, 512, np.array([.07, .07, .08]), np.array([.78, .77, .74]), 2.5, 5)).save(b)
    return {"W": w, "B": b}
MARBLE_DENSITY = 2700                              # 大理石の密度(kg/m^3)。一辺5cmで約0.34kg
def torus_mesh():                                  # 見た目用のなめらかなトーラス(当たり判定は別のカプセル列)
    os.makedirs(TEX_DIR, exist_ok=True); fn = f"{TEX_DIR}/torus.obj"
    if os.path.exists(fn): return fn
    N, M, L = 192, 48, ["# torus"]
    for i in range(N+1):
        th = 2*np.pi*i/N
        for j in range(M+1):
            ph = 2*np.pi*j/M; c, s = np.cos(ph), np.sin(ph)
            L += [f"v {(RING_R+RING_r*c)*np.cos(th):.6f} {(RING_R+RING_r*c)*np.sin(th):.6f} {RING_r*s:.6f}",
                  f"vn {c*np.cos(th):.6f} {c*np.sin(th):.6f} {s:.6f}", f"vt {8*i/N:.6f} {j/M:.6f}"]
    k = lambda i, j: i*(M+1)+j+1
    for i in range(N):
        for j in range(M):
            a, b, c2, d = k(i, j), k(i+1, j), k(i+1, j+1), k(i, j+1)
            L += [f"f {a}/{a}/{a} {b}/{b}/{b} {c2}/{c2}/{c2}", f"f {a}/{a}/{a} {c2}/{c2}/{c2} {d}/{d}/{d}"]
    open(fn, "w").write("\n".join(L)); return fn
def _mat(color, name):
    m = CustomMaterial(texture="WoodLight", tex_name=name, mat_name=name+"_mat", tex_attrib={"type": "cube"},
                       mat_attrib={"specular": "0.4", "shininess": "0.5", "texrepeat": "1 1"})
    m.tex_attrib["file"] = textures()[color]; return m
def spot_xy(deg): t = np.radians(deg); return RING_C + SPOT_R*np.array([np.cos(t), np.sin(t)])
def lookat_quat(pos, target):                     # MJCFカメラ向きの四元数(w,x,y,z)
    f = np.asarray(target, float) - pos; f /= np.linalg.norm(f); z = -f
    x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x); y = np.cross(z, x); R = np.c_[x, y, z]
    w = np.sqrt(max(1+np.trace(R), 1e-9))/2
    return [w, (R[2, 1]-R[1, 2])/(4*w), (R[0, 2]-R[2, 0])/(4*w), (R[1, 0]-R[0, 1])/(4*w)]
class RingTower(Stack):
    def _load_model(self):
        super(Stack, self)._load_model()
        self.robots[0].robot_model.set_base_xpos(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        tgt = [RING_C[0], RING_C[1], TABLE_Z + 0.11]
        for nm, pos in (("artview", [0.62, -0.58, 1.22]), ("sideview", [RING_C[0], -0.85, 1.05])):
            arena.set_camera(camera_name=nm, pos=pos, quat=lookat_quat(np.array(pos), tgt))
        arena.set_camera(camera_name="topcam", pos=[RING_C[0], RING_C[1], TOPCAM_Z], quat=[1, 0, 0, 0],
                         camera_attribs={"fovy": str(TOPCAM_FOVY)})           # 真上から見下ろす深度カメラ(知覚用)
        ET.SubElement(arena.asset, "texture", name="ringtex", type="2d", file=textures()["W"])
        ET.SubElement(arena.asset, "material", name="ringmat", texture="ringtex", specular="0.35", shininess="0.5")
        ET.SubElement(arena.asset, "mesh", name="torus", file=torus_mesh())
        body = ET.Element("body", name="ring", pos=f"{RING_C[0]} {RING_C[1]} {TABLE_Z + RING_r}")
        for i in range(NSEG):
            t0, t1 = 2*np.pi*i/NSEG, 2*np.pi*(i+1)/NSEG
            p = np.r_[RING_R*np.cos(t0), RING_R*np.sin(t0), 0, RING_R*np.cos(t1), RING_R*np.sin(t1), 0]
            ET.SubElement(body, "geom", name=f"ring_seg{i}", type="capsule", size=f"{RING_r}",
                          fromto=" ".join(f"{v:.5f}" for v in p), group="3")          # 当たり判定(描画しない)
        ET.SubElement(body, "geom", name="ring_visual", type="mesh", mesh="torus", material="ringmat",
                      contype="0", conaffinity="0", group="1")                                   # 見た目(当たり判定なし)
        arena.worldbody.append(body)
        self.cubes = [BoxObject(name=f"cube{i+1}", size_min=[H]*3, size_max=[H]*3, rgba=None, material=_mat(c, f"marble{i+1}"), density=MARBLE_DENSITY,
                                solref=(0.004, 1), solimp=(0.99, 0.99, 0.001))       # 大理石を想定した硬い接触(verify.pyと同じ)
                      for i, c in enumerate(COLORS)]
        self.cubeB, self.cubeA = self.cubes[0], self.cubes[1]   # Stack互換(観測cubeA/cubeB)
        s = SequentialCompositeSampler(name="ObjectSampler")
        kw = dict(rotation=None, ensure_object_boundary_in_range=False, ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=0.01)
        for i, obj in enumerate(self.cubes):
            x, y = spot_xy(SPOT_DEG[i])
            s.append_sampler(UniformRandomSampler(name=f"s{i}", mujoco_objects=obj, x_range=[x-0.01, x+0.01], y_range=[y-0.01, y+0.01], **kw))
        self.placement_initializer = s
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=self.cubes)
    def pose(self, i):                            # i番目(0始まり)のキューブの位置・四元数(x,y,z,w)
        bid = self.sim.model.body_name2id(self.cubes[i].root_body)
        q = self.sim.data.body_xquat[bid]; return self.sim.data.body_xpos[bid].copy(), np.r_[q[1:], q[0]]
def make_tower_env(cams=("agentview", "artview"), size=256):
    import robosuite as suite
    return RingTower(robots="Panda", controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, camera_names=list(cams),
        camera_heights=size, camera_widths=size, control_freq=20, horizon=5000, ignore_done=True)
