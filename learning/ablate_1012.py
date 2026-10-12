# 切り分け実験(10/12)の一括実行。v7 の教材(dagger_r0〜r5, dagger_eval4)をそのまま使い、入力と教材を変えて学習し直し、
# 同じ未知44場面(種 70000〜71003、README 記載)で生徒単独の成功数を比べる。最初に pol_r5 を評価して 43/44 の再現を確かめる。
import os, sys, glob, json, shutil, subprocess, datetime, numpy as np
from zoneinfo import ZoneInfo
W = os.path.dirname(os.path.abspath(__file__)); M, PY = "/content/mk7", "/content/ev310/bin/python"
DRV = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "bc_ablate_1012"); os.makedirs(DRV, exist_ok=True)
NW = max(1, (os.cpu_count() or 2) - 1)
SEEDS = [70000 + 100*w + i for w in range(11) for i in range(4)]                      # pol_r5 の評価と同じ44場面
def log(m):
    s = f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%m/%d %H:%M:%S')}] {m}"; print(s, flush=True)
    open(f"{DRV}/ablate.log", "a").write(s + "\n")
def evaluate(name, pol):
    out = f"{M}/abl_eval/{name}"; os.makedirs(out, exist_ok=True)
    parts = [SEEDS[i::NW] for i in range(NW)]
    ps = [subprocess.Popen([PY, "-u", "ablate_worker.py", pol, out, ",".join(map(str, p))], cwd=W,
                           stdout=open(f"{out}/w{i}.log", "w"), stderr=subprocess.STDOUT) for i, p in enumerate(parts) if p]
    for p in ps: p.wait()
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/ep*.json"))]
    c = [r["ok"] for r in R if r["seed"] % 2 == 0]; n = [r["ok"] for r in R if r["seed"] % 2 == 1]
    shutil.copytree(out, f"{DRV}/eval/{name}", dirs_exist_ok=True)
    return dict(name=name, total=sum(c) + sum(n), n=len(R), clean=f"{sum(c)}/{len(c)}", noisy=f"{sum(n)}/{len(n)}")
D = [f"{M}/dagger_r{i}" for i in range(6)] + ([f"{M}/dagger_eval4"] if os.path.isdir(f"{M}/dagger_eval4") else [])
CONDS = [("full", D, "all"), ("notilt", D, "notilt"), ("nomm", D, "nomm"), ("noextra", D, "noextra"), ("bconly", D[:1], "all")]
SUM = f"{DRV}/summary.json"; S = json.load(open(SUM)) if os.path.exists(SUM) else []
S = [s for s in S if s["n"] == len(SEEDS)]                                      # 欠けのある評価は数え直す(終わった場面は飛ばす)
done = {s["name"] for s in S}
log(f"開始: 並列 {NW}、教材 {[os.path.basename(d) for d in D]}")
# 10/12: 並列の担当が同時に円環の形(/tmp/pai_tex/torus.obj)を書き出して読み損ねる競合があったため、先に1回だけ作っておく
subprocess.run([PY, "-c", "import sys; sys.path[:0]=['../sim','.']; from tower_env import textures, torus_mesh; textures(); torus_mesh()"], cwd=W)
if "control_pol_r5" not in done:
    r = evaluate("control_pol_r5", f"{M}/pol_r5.npz"); S.append(r); json.dump(S, open(SUM, "w"), indent=1); log(f"対照 {r}")
    if r["total"] != 43 or r["n"] != len(SEEDS): log("pol_r5 が 43/44 を再現しないため停止(環境が10/4と違う可能性)"); log("ABL_DONE"); sys.exit()
for cname, dirs, keep in CONDS:
    for seed in (0, 1, 2):
        name = f"{cname}_s{seed}"
        if name in done: continue
        pol = f"{M}/abl_{name}.npz"
        t = subprocess.run(["python3", "ablate_train.py", ",".join(dirs), pol, keep, str(seed)], cwd=W, capture_output=True, text=True)
        open(f"{DRV}/train_{name}.log", "w").write(t.stdout + t.stderr)
        if t.returncode: log(f"{name} 学習失敗: {t.stderr[-300:]}"); continue
        shutil.copy(pol, DRV); r = evaluate(name, pol); S.append(r); json.dump(S, open(SUM, "w"), indent=1); log(f"{name} {r['total']}/{r['n']} (乱れなし {r['clean']}, 乱れあり {r['noisy']})")
log("ABL_DONE")
