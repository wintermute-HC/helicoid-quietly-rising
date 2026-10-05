import os, subprocess, glob, json, shutil, datetime, numpy as np
from zoneinfo import ZoneInfo
W, M, PY = os.path.dirname(os.path.abspath(__file__)), "/content/mk7", "/content/ev310/bin/python"; DRV = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "bc_v7")  # W=このフォルダ(learning/), DRV=成果の保存先(環境変数 HELICOID_DRIVE)
NW = max(1, (os.cpu_count() or 2) - 1); ts = lambda: datetime.datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%H:%M:%S")
dirs = [f"{M}/dagger_r{i}" for i in range(6)] + [f"{M}/dagger_eval4"]
print(f"[{ts()}] 学習 pol_r5 開始", flush=True)
t = subprocess.run(["python3", "train_bc.py", ",".join(dirs), f"{M}/pol_r5.npz", "40"], cwd=W, capture_output=True, text=True)
open(f"{M}/train_r5.log", "w").write(t.stdout + t.stderr)
print(f"[{ts()}] 学習 pol_r5: {[l for l in t.stdout.splitlines() if l.startswith('{')][-1:]} rc={t.returncode}", flush=True)
if t.returncode: print(t.stderr[-800:], flush=True); print("R5_DONE", flush=True); raise SystemExit
shutil.copy(f"{M}/pol_r5.npz", DRV); shutil.copy(f"{M}/train_r5.log", DRV)
out = f"{M}/eval_r5"; print(f"[{ts()}] 評価開始 pol_r5 回数 {NW*4}", flush=True)
procs = [subprocess.Popen([PY, "-u", "mk_worker.py", f"{M}/pol_r5.npz", "0.0", "0.05", out,
         ",".join(str(70000 + 100*w + i) for i in range(4))], cwd=W, stdout=open(f"{M}/ev5_w{w}.log", "w"), stderr=subprocess.STDOUT) for w in range(NW)]
for p in procs: p.wait()
E = {int(f[-9:-4]): bool(np.load(f)["ok"]) for f in sorted(glob.glob(f"{out}/ep*.npz"))}
c = [v for s, v in E.items() if s % 2 == 0]; n = [v for s, v in E.items() if s % 2 == 1]
S = dict(policy="pol_r5", total=f"{sum(E.values())}/{len(E)}", clean=f"{sum(c)}/{len(c)}", noisy=f"{sum(n)}/{len(n)}")
json.dump(S, open(f"{M}/eval_r5_summary.json", "w"), ensure_ascii=False); shutil.copy(f"{M}/eval_r5_summary.json", DRV)
for f in glob.glob(f"{M}/ev5_w*.log"): shutil.copy(f, DRV)
print(f"[{ts()}] 評価完了 {S}", flush=True); print("R5_DONE", flush=True)
