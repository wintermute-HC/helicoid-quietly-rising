import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import json, os; os.environ["MUJOCO_GL"]="egl"
from composer import compose
plans = [{"title":"静昇","concept":"静けさを一定の小さな回転で表す","critique":"","levels":[{"rel_deg":0}]+[{"rel_deg":18}]*4,"final":False},
         {"title":"静昇","concept":"累積90°の四分の一回転","critique":"18°刻みでは累積72°で、上端が下端とほぼ同じ向きに戻って見える。立方体の90°対称性のため。22.5°にして累積を四分の一回転にそろえる。","levels":[{"rel_deg":0}]+[{"rel_deg":22.5}]*4,"final":False},
         {"title":"静昇","concept":"累積90°の四分の一回転","critique":"予想図で、ねじれが一様に立ち上がるのを確認した。確定する。","levels":[{"rel_deg":0}]+[{"rel_deg":22.5}]*4,"final":True}]
it = iter(plans)
d, tr = compose("静かに立ち上がる螺旋", propose=lambda m: "```json\n"+json.dumps(next(it), ensure_ascii=False)+"\n```", log_path="mockrun/log.json", img_dir="mockrun")
print("mock", len(tr))
