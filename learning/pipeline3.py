import os, subprocess, glob, json, datetime, shutil, tarfile
from zoneinfo import ZoneInfo
import numpy as np
W, M, PY = os.path.dirname(os.path.abspath(__file__)), "/content/mk7", "/content/ev310/bin/python"; DRV = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "bc_v7")  # W=このフォルダ(learning/), DRV=成果の保存先(環境変数 HELICOID_DRIVE)
os.makedirs(M, exist_ok=True); os.makedirs(DRV, exist_ok=True); NW = max(1, (os.cpu_count() or 2) - 1)
os.environ.pop("TEACHER", None)                                   # 先生は markov_teacher.py(漏斗型は使わない)
def log(m):
    s = f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%m/%d %H:%M:%S JST')}] {m}"
    print(s, flush=True); open(f"{M}/pipeline.log", "a").write(s + "\n")
def save(summary):
    json.dump(summary, open(f"{M}/summary.json", "w"), ensure_ascii=False, indent=1)
    for f in glob.glob(f"{M}/pol_*.npz") + glob.glob(f"{M}/*.log") + [f"{M}/summary.json"]: shutil.copy(f, DRV)
ROUNDS = [("r0", None, 1.0, 0.05, 4), ("r1", "pol_r0", 0.5, 0.0, 2), ("r2", "pol_r1", 0.2, 0.0, 2),
          ("r3", "pol_r2", 0.0, 0.0, 2), ("r4", "pol_r3", 0.0, 0.0, 2), ("r5", "pol_r4", 0.0, 0.0, 2)]
log(f"開始 v7: 並列 {NW} プロセス"); dirs, summary = [], []
for name, pol, beta, sigma, per in ROUNDS:
    out = f"{M}/dagger_{name}"; r = int(name[1:]); log(f"{name} 開始 (生徒の割合 {1-beta:.0%}, 方策 {pol}, 回数 {NW*per})"); procs = []
    for w in range(NW):
        seeds = ",".join(str(10000*r + 100*w + i) for i in range(per))
        procs.append(subprocess.Popen([PY, "-u", "mk_worker.py", f"{M}/{pol}.npz" if pol else "none", str(beta), str(sigma), out, seeds],
                                      cwd=W, stdout=open(f"{M}/{name}_w{w}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs: p.wait()
    files = sorted(glob.glob(f"{out}/ep*.npz")); E = [np.load(f) for f in files]
    ok = sum(bool(e["ok"]) for e in E); smp = sum(len(e["X"]) for e in E)
    rec = dict(round=name, student_share=1-beta, ok=ok, n=len(E), samples=smp)
    log(f"{name} 完了: 成功 {ok}/{len(E)}, 教材 {smp} 件")
    if name == "r0":
        ev = [e for f, e in zip(files, E) if int(f[-9:-4]) % 2 == 0]; nz = [e for f, e in zip(files, E) if int(f[-9:-4]) % 2 == 1]
        okc, okn = sum(bool(e["ok"]) for e in ev), sum(bool(e["ok"]) for e in nz); rec.update(clean_ok=okc, noisy_ok=okn)
        log(f"r0 内訳: 乱れなし {okc}/{len(ev)}, 乱れあり {okn}/{len(nz)}")
        if okc*2 < len(ev):
            log("先生の成功が半数未満のため停止"); summary.append(rec); save(summary); log("ALL_DONE"); raise SystemExit
    summary.append(rec); dirs.append(out)
    if name != ROUNDS[-1][0]:
        t = subprocess.run(["python3", "train_bc.py", ",".join(dirs), f"{M}/pol_{name}.npz", "40"], cwd=W, capture_output=True, text=True)
        open(f"{M}/train_{name}.log", "w").write(t.stdout + t.stderr)
        last = [l for l in t.stdout.splitlines() if l.startswith("{")][-1:]
        log(f"学習 pol_{name}: {last} rc={t.returncode}")
        if t.returncode: log("学習失敗のため中止: " + t.stderr[-500:]); save(summary); break
    save(summary)
with tarfile.open(f"{DRV}/dagger_data_v7.tar.gz", "w:gz") as tf:
    for d in dirs: tf.add(d, arcname=os.path.basename(d))
log("教材をDriveに保存: dagger_data_v7.tar.gz"); save(summary); log("ALL_DONE")
