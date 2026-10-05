# 円環(トーラス)を衝突体として置き、キューブを円環の外から中央へ運ぶ環境
import numpy as np, xml.etree.ElementTree as ET
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from twist_env import H, _mat
RING_C = np.array([-0.05, 0.0])     # 円環中心(テーブル座標 x,y)
RING_R, RING_r, NSEG = 0.16, 0.025, 32   # 中心線半径・管半径・近似カプセル数
SPOT_R = 0.27                        # キューブ初期位置の半径(円環中心から)
SPOTS = {"B": 125.0, "A": -125.0}    # 初期位置の方位角(度、+x基準)
def spot_xy(deg): t = np.radians(deg); return RING_C + SPOT_R*np.array([np.cos(t), np.sin(t)])
class RingStack(Stack):
    def _load_model(self):
        super(Stack, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        arena.set_camera(camera_name="topview", pos=[RING_C[0], RING_C[1], 1.55], quat=[1, 0, 0, 0])
        # 円環: 静止体、カプセルを円周に並べて衝突形状を近似
        zt = self.table_offset[2] + RING_r
        body = ET.Element("body", name="ring", pos=f"{RING_C[0]} {RING_C[1]} {zt}")
        for i in range(NSEG):
            t0, t1 = 2*np.pi*i/NSEG, 2*np.pi*(i+1)/NSEG
            p0 = RING_R*np.array([np.cos(t0), np.sin(t0), 0]); p1 = RING_R*np.array([np.cos(t1), np.sin(t1), 0])
            ET.SubElement(body, "geom", name=f"ring_seg{i}", type="capsule", size=f"{RING_r}",
                          fromto=" ".join(f"{v:.5f}" for v in np.r_[p0, p1]), rgba="0.92 0.92 0.9 1",
                          friction="1 0.005 0.0001", group="1")
        arena.worldbody.append(body)
        self.cubeA = BoxObject(name="cubeA", size_min=[H]*3, size_max=[H]*3, rgba=[1,0,0,1], material=_mat("WoodRed","redwood"))
        self.cubeB = BoxObject(name="cubeB", size_min=[H]*3, size_max=[H]*3, rgba=[0,1,0,1], material=_mat("WoodGreen","greenwood"))
        s = SequentialCompositeSampler(name="ObjectSampler")
        kw = dict(rotation=None, ensure_object_boundary_in_range=False, ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=0.01)
        for nm, obj in (("B", self.cubeB), ("A", self.cubeA)):
            x, y = spot_xy(SPOTS[nm])
            s.append_sampler(UniformRandomSampler(name=nm, mujoco_objects=obj, x_range=[x-0.01, x+0.01], y_range=[y-0.01, y+0.01], **kw))
        self.placement_initializer = s
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=[self.cubeA, self.cubeB])
def make_ring_env(cams=("agentview", "robot0_eye_in_hand", "topview"), size=256):
    import robosuite as suite
    return RingStack(robots="Panda", controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, camera_names=list(cams),
        camera_heights=size, camera_widths=size, control_freq=20, horizon=2000, ignore_done=True)
