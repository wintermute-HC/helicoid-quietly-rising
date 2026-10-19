# 10/9: 映像の2倍速区間を決めるため、撮影と同じ場面(seed 70300、生徒 pol_r5 単独)を画面を描かずに再シミュレーションし、
#       各石を運ぶ前の観測の時点(歩数)を記録する。歩数が撮影時の1,897と一致することを確認する。
# 出力: marks1009.json(fa = 2個目の石の観測の時点、fb = 5個目の石の観測の時点。いずれも main.mp4 のフレーム番号 = 冒頭静止40 + 歩数)
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os; os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import json, numpy as np, robosuite as suite
from tower_env import RingTower
from test_markov import MarkovTower
from learn_common import NumpyMLP
SEED = 70300; REL = [0, 22.5, 22.5, 22.5, 22.5]
POL = os.environ.get('POL', os.path.join(_R, 'policy', 'pol_r5.npz')); OUT = os.environ.get('MARKS', 'marks1009.json')
env = RingTower(robots="Panda", controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
                has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, control_freq=20, horizon=8000, ignore_done=True)
np.random.seed(SEED); o = env.reset()
ex = MarkovTower(env, REL, pol=NumpyMLP(POL), beta=0.0, seed=SEED)
n = 0; looks = []; orig = ex._look
def lk():
    orig(); looks.append(dict(step=n, k=int(ex.k)))
ex._look = lk
while not ex.done and n < 6000 and ex.stuck < 3:
    o, *_ = env.step(ex.act(o)); n += 1
k1 = [l['step'] for l in looks if l['k'] == 1]; k4 = [l['step'] for l in looks if l['k'] == 4]
M = dict(steps=n, fa=40 + k1[0], fb=40 + k4[0], looks=looks)
json.dump(M, open(OUT, 'w')); print('MARKS', json.dumps(M), flush=True)
if n != 1897: raise SystemExit(f'歩数が撮影と不一致: {n}')
# 10/9の実行結果: steps=1897, fa=658, fb=1543
