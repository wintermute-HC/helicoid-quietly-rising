# DARTデータ(npz) → LeRobot形式。観測の作り方は学習済みモデルと同一(画像180°反転、state=位置+軸角(xyzw)+グリッパ)
import glob, os, sys, shutil, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
SRC = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "gen_dart_v1")
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/content/ds/twist_dart_v1"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
def quat2aa(q):
    x, y, z, w = q; w = np.clip(w, -1, 1); den = np.sqrt(max(1-w*w, 0))
    return np.zeros(3) if den < 1e-8 else np.array([x, y, z])/den*2*np.arccos(w)
IMG = {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
F = {"observation.images.image": IMG, "observation.images.image2": IMG,
     "observation.state": {"dtype": "float32", "shape": (8,), "names": ["x","y","z","ax","ay","az","g0","g1"]},
     "action": {"dtype": "float32", "shape": (7,), "names": ["dx","dy","dz","dax","day","daz","grip"]}}
if os.path.exists(ROOT): shutil.rmtree(ROOT)
ds = LeRobotDataset.create(repo_id="local/twist_dart_v1", fps=20, features=F, root=ROOT, robot_type="panda",
                           use_videos=True, image_writer_threads=8, vcodec="h264")
files = sorted(glob.glob(f"{SRC}/*.npz"))[:LIMIT or None]
for i, f in enumerate(files):
    z = dict(np.load(f)); task = str(z["instruction"])
    for t in range(len(z["action"])):
        ds.add_frame({"observation.images.image":  np.ascontiguousarray(z["agentview"][t][::-1, ::-1]),
                      "observation.images.image2": np.ascontiguousarray(z["wrist"][t][::-1, ::-1]),
                      "observation.state": np.r_[z["eef_pos"][t], quat2aa(z["eef_quat"][t]), z["grip_qpos"][t]].astype(np.float32),
                      "action": z["action"][t].astype(np.float32), "task": task})
    ds.save_episode()
    print(f"EP {i+1}/{len(files)} {os.path.basename(f)} T={len(z['action'])} | {task}", flush=True)
ds.finalize(); print("CONVERT_DONE", flush=True)
