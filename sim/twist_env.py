import numpy as np
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
H = 0.025  # 半辺 → 5cm角
def _mat(tex, n):
    return CustomMaterial(texture=tex, tex_name=n, mat_name=n+"_mat", tex_attrib={"type":"cube"},
                          mat_attrib={"texrepeat":"1 1","specular":"0.4","shininess":"0.1"})
class TwistStack(Stack):
    def _load_model(self):
        super(Stack, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0,0,0])
        arena.set_camera(camera_name="topview", pos=[0,0,1.25], quat=[1,0,0,0])
        self.cubeA = BoxObject(name="cubeA", size_min=[H]*3, size_max=[H]*3, rgba=[1,0,0,1], material=_mat("WoodRed","redwood"))
        self.cubeB = BoxObject(name="cubeB", size_min=[H]*3, size_max=[H]*3, rgba=[0,1,0,1], material=_mat("WoodGreen","greenwood"))
        s = SequentialCompositeSampler(name="ObjectSampler")
        kw = dict(rotation=None, ensure_object_boundary_in_range=False, ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=0.01)
        s.append_sampler(UniformRandomSampler(name="B", mujoco_objects=self.cubeB, x_range=[-0.02,0.02], y_range=[0.06,0.10], **kw))
        s.append_sampler(UniformRandomSampler(name="A", mujoco_objects=self.cubeA, x_range=[-0.03,0.03], y_range=[-0.12,-0.08], **kw))
        self.placement_initializer = s
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=[self.cubeA, self.cubeB])
def quat2mat(q):  # robosuite quat = (x,y,z,w)
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def yaw_of(q):
    R = quat2mat(q); return np.arctan2(R[1,0], R[0,0])
def wrap90(a): return (a+np.pi/4) % (np.pi/2) - np.pi/4   # 正方形の90°対称性
def make_env(cams=("agentview","robot0_eye_in_hand","topview"), size=256):
    import robosuite as suite
    return TwistStack(robots="Panda", controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, camera_names=list(cams),
        camera_heights=size, camera_widths=size, control_freq=20, horizon=1000, ignore_done=True)
