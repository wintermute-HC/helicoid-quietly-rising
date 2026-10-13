# 停止の位置の特定(10/13): 切り分け実験の方策を、同じ未知44場面でもう一度動かし、歩ごとに「いまどの局面か」と
# 掴む位置・置く位置までの距離、指の指令(生徒と先生)を記録する。停止(1個の運動が600歩で終わらない)が起きた局面を数える。
# 方策の動きは ablate_worker.py と同じ(乱数の使い方も同じ)なので、成否と歩数は 10/12 の評価と一致するはず。それを照合する。
# 使い方: python3 learning/stall_1013.py [方策名,...]   (既定: full_s0,notilt_s0,nomm_s0)
#         内部で python stall_1013.py --worker <方策npz> <出力フォルダ> <種,...> を並列に起動する
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, sys, json, time, glob, subprocess, datetime
PHASES = ["接近", "掴む直前", "閉じる途中", "持ち上げ", "運搬・回転", "置く直前", "離した後"]
SEEDS = [70000 + 100*w + i for w in range(11) for i in range(4)]                     # 10/12 と同じ44場面

def worker(pol_path, out, seeds):
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as np
    from tower_env import make_tower_env
    from tower_expert import tower_report_by_height as tower_report
    from twist_env import wrap90
    from learn_common import Tracker, features, act_done, eef, gap
    from markov_teacher import teacher
    from learn_common import NumpyMLP, F_DIM
    from test_markov import MarkovTower, REL, MAX_ACT
    class Masked(NumpyMLP):                                                       # ablate_worker.py と同じ
        def __init__(self, path):
            super().__init__(path); z = np.load(path); self.keep = z["keep"] if "keep" in z.files else np.arange(F_DIM)
        def __call__(self, f): return super().__call__(np.asarray(f)[self.keep])
    class Logged(MarkovTower):
        """MarkovTower.act と同じ手順(乱数の消費順も同じ)に、記録だけを足したもの"""
        def __init__(self, *a, **k):
            super().__init__(*a, **k); self.trace, self.stall_events = [], []
        def act(self, o):
            if self.stage != "act" or self.done: return super(MarkovTower, self).act(o)
            if self.tr is None: self.tr = Tracker(self.P0, self.yP0); self.na = 0
            self.tr.update(o)
            if act_done(o, self.tr, self.D) or self.na >= MAX_ACT:
                if self.na >= MAX_ACT: self.stuck += 1; self.event()
                self.k += 1; self.stage = "view"; self.t = 0; self.tr = None; return super(MarkovTower, self).act(o)
            f = features(o, self.tr, self.axis, self.D, self.ygoal); at = teacher(o, self.tr, self.D, self.ygoal)
            self.X.append(f); self.A.append(at.copy()); self.na += 1
            a = at if (self.pol is None or self.rng.random() < self.beta) else self.pol(f)
            if self.noise > 0: a = a.copy(); a[:6] = np.clip(a[:6] + min(1.0, float(np.abs(at[:3]).max()))*self.rng.normal(0, self.noise, 6), -1, 1)
            self.record(o, at, a)
            self.tr.prev_g = a[6]; return a
        def record(self, o, at, a):
            tr = self.tr; e, _ = eef(o); held = tr.off is not None
            dg = float(np.linalg.norm(tr.S + np.array([0, 0, 0.005]) - e))
            dp = float(np.linalg.norm(np.r_[self.D[:2] - tr.S[:2], self.D[2] + 0.003 - tr.S[2]]))
            dxy = float(np.linalg.norm(self.D[:2] - tr.S[:2])); yerr = abs(wrap90(self.ygoal - tr.yS))
            if not held and not tr.held_once: ph = 2 if tr.prev_g > 0 else (1 if dg < 0.01 else 0)
            elif held: ph = 3 if not tr.lifted else (4 if (dxy > 0.008 or yerr > 0.02) else 5)
            else: ph = 6
            self.trace.append((self.k, self.na, ph, dg, dp, gap(o), float(at[6]), float(a[6]), float(np.linalg.norm(a[:3])), float(np.linalg.norm(at[:3]))))
        def event(self):
            L = [r for r in self.trace if r[0] == self.k][-100:]
            if not L: self.stall_events.append(dict(k=self.k)); return
            ph = [r[2] for r in L]; last = L[-1]
            self.stall_events.append(dict(k=int(self.k), last_phase=PHASES[last[2]], phases_last100={PHASES[p]: ph.count(p) for p in set(ph)},
                                    dg_mm=round(last[3]*1000, 2), dp_mm=round(last[4]*1000, 2), min_dg_mm=round(min(r[3] for r in L)*1000, 2),
                                    min_dp_mm=round(min(r[4] for r in L)*1000, 2), gap_mm=round(last[5]*1000, 1),
                                    grip_disagree_last100=sum(1 for r in L if (r[6] > 0) != (r[7] > 0)),
                                    student_move=round(float(np.mean([r[8] for r in L])), 4), teacher_move=round(float(np.mean([r[9] for r in L])), 4),
                                    student_g_last=last[7], teacher_g_last=last[6]))
    pol = Masked(pol_path); os.makedirs(out, exist_ok=True)
    for s in seeds:
        fn = f"{out}/ep{s:05d}.json"
        if os.path.exists(fn): continue
        t0 = time.time(); env = make_tower_env(cams=("agentview",), size=32); np.random.seed(s); o = env.reset()
        ex = Logged(env, REL, pol=pol, beta=0.0, noise=(0.05 if s % 2 else 0.0), seed=s); n = 0
        while not ex.done and n < 6000 and ex.stuck < 3: o, *_ = env.step(ex.act(o)); n += 1
        for _ in range(60): o, *_ = env.step(np.r_[np.zeros(6), -1.0])
        ok, rows = tower_report(env, REL)
        np.savez_compressed(f"{out}/ep{s:05d}.npz", log=np.array(ex.trace, np.float32),
                            cols=np.array(["k", "na", "phase", "dg", "dp", "gap", "g_teacher", "g_student", "move_student", "move_teacher"]))
        r = dict(seed=s, ok=bool(ok), steps=n, stuck=ex.stuck, sec=round(time.time() - t0), events=ex.stall_events)
        json.dump(r, open(fn, "w"), ensure_ascii=False); print(json.dumps(r, ensure_ascii=False), flush=True)
        try: env.close()
        except Exception: pass

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    worker(sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")]); sys.exit()

if __name__ == "__main__":
    from zoneinfo import ZoneInfo
    W = os.path.dirname(os.path.abspath(__file__)); PY = "/content/ev310/bin/python"
    SRC = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "bc_ablate_1012")
    DRV = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "stall_1013"); os.makedirs(DRV, exist_ok=True)
    NAMES = (sys.argv[1] if len(sys.argv) > 1 else "full_s0,notilt_s0,nomm_s0").split(",")
    NW = max(1, (os.cpu_count() or 2) - 1)
    def log(m):
        s = f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%m/%d %H:%M:%S')}] {m}"; print(s, flush=True)
        open(f"{DRV}/stall.log", "a").write(s + "\n")
    log(f"開始: 並列 {NW}、方策 {NAMES}")
    subprocess.run([PY, "-c", "import sys; sys.path[:0]=['../sim','.']; from tower_env import textures, torus_mesh; textures(); torus_mesh()"], cwd=W)
    for name in NAMES:
        out = f"{DRV}/{name}"; os.makedirs(out, exist_ok=True); pol = f"{SRC}/abl_{name}.npz"
        todo = [s for s in SEEDS if not os.path.exists(f"{out}/ep{s:05d}.json")]
        parts = [todo[i::NW] for i in range(NW)]
        ps = [subprocess.Popen([PY, "-u", "stall_1013.py", "--worker", pol, out, ",".join(map(str, p))], cwd=W,
                               stdout=open(f"{out}/w{i}.log", "a"), stderr=subprocess.STDOUT) for i, p in enumerate(parts) if p]
        for p in ps: p.wait()
        R = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/ep*.json"))]
        ref = {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob(f"{SRC}/eval/{name}/ep*.json")}
        same = sum(1 for r in R if r["seed"] in ref and ref[r["seed"]]["ok"] == r["ok"] and ref[r["seed"]]["steps"] == r["steps"])
        log(f"{name}: {len(R)}場面、成功 {sum(r['ok'] for r in R)}、停止 {sum(r['stuck'] for r in R)}回、10/12と成否・歩数が一致 {same}/{len(R)}")
    log("STALL_DONE")
