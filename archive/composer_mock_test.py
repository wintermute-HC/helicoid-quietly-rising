import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import json, os; os.environ["MUJOCO_GL"]="egl"
from composer import compose
plans = [{"title":"息","concept":"試験","critique":"","levels":[{"rel_deg":0},{"rel_deg":10},{"rel_deg":20},{"rel_deg":10},{"rel_deg":0}],"final":False},
         {"title":"息","concept":"試験","critique":"予想図を確認","levels":[{"rel_deg":0},{"rel_deg":10},{"rel_deg":20},{"rel_deg":10},{"rel_deg":0}],"final":True}]
it = iter(plans)
def mock(msgs):
    if len(msgs) > 1: c = msgs[-1]["content"]; print("  ← Claudeへ:", [b["type"] for b in c], "画像サイズ", len(c[0]["source"]["data"]))
    return "```json\n" + json.dumps(next(it), ensure_ascii=False) + "\n```"
d, tr = compose("静かな呼吸", propose=mock, img_dir="/tmp/mockimg"); print("確定:", d["title"], len(tr), "回")
