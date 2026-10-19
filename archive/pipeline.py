import os, subprocess, glob, json, datetime, shutil
from zoneinfo import ZoneInfo
import numpy as np
W, M, PY = "/content/pai", "/content/mk", "/content/ev310/bin/python"; DRV = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "bc_v5")
os.makedirs(M, exist_ok=True); os.makedirs(DRV, exist_ok=True); NW = max(1, (os.cpu_count() or 2) - 1)
def log(m):
    s = f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%m/%d %H:%M:%S JST')}] {m}"
    print(s, flush=True); open(f"{M}/pipeline.log", "a").write(s + "\n")
def save(summary):
    json.dump(summary, open(f"{M}/summary.json", "w"), ensure_ascii=False, indent=1)
    for f in glob.glob(f"{M}/pol_*.npz") + glob.glob(f"{M}/*.log") + [f"{M}/summary.json"]: shutil.copy(f, DRV)
ROUNDS = [("r0", None, 1.0, 0.15, 4), ("r1", "pol_r0", 0.5, 0.0, 2), ("r2", "pol_r1", 0.2, 0.0, 2), ("r3", "pol_r2", 0.0, 0.0, 2), ("r4", "pol_r3", 0.0, 0.0, 2)]
log(f"開始: 並列 {NW} プロセス"); dirs, summary = [], []
for name, pol, beta, sigma, per in ROUNDS:
    out = f"{M}/dagger_{name}"; r = int(name[1:]); log(f"{name} 開始 (生徒の割合 {1-beta:.0%}, 方策 {pol}, 回数 {NW*per})"); procs = []
    for w in range(NW):
        seeds = ",".join(str(10000*r + 100*w + i) for i in range(per))
        procs.append(subprocess.Popen([PY, "-u", "mk_worker.py", f"{M}/{pol}.npz" if pol else "none", str(beta), str(sigma), out, seeds],
                                      cwd=W, stdout=open(f"{M}/{name}_w{w}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs: p.wait()
    E = [np.load(f) for f in sorted(glob.glob(f"{out}/ep*.npz"))]
    ok = sum(bool(e["ok"]) for e in E); smp = sum(len(e["X"]) for e in E)
    log(f"{name} 完了: 成功 {ok}/{len(E)}, 教材 {smp} 件"); summary.append(dict(round=name, student_share=1-beta, ok=ok, n=len(E), samples=smp)); dirs.append(out)
    if name != "r4":
        t = subprocess.run(["python3", "train_bc.py", ",".join(dirs), f"{M}/pol_{name}.npz", "40"], cwd=W, capture_output=True, text=True)
        open(f"{M}/train_{name}.log", "w").write(t.stdout + t.stderr)
        last = [l for l in t.stdout.splitlines() if l.startswith("{")][-1:]
        log(f"学習 pol_{name}: {last} rc={t.returncode}")
        if t.returncode: log("学習失敗のため中止: " + t.stderr[-500:]); save(summary); break
    save(summary)
log("ALL_DONE")
