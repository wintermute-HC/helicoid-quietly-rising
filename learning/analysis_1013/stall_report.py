# 停止の位置の集計(stall_1013.py の出力から)。論文 4.5節の表D・表E。
import glob, json, os, collections, datetime, numpy as np
B = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "stall_1013")
print("JST", datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%m/%d %H:%M"))
PH = ["接近", "掴む直前", "閉じる途中", "持ち上げ", "運搬・回転", "置く直前", "離した後"]
print("## 表S1 停止が起きた局面(停止の回数)")
print("| 方策 | 場面 | 成功 | 停止 | " + " | ".join(PH) + " |"); print("|---|---|---|---|" + "---|"*len(PH))
EV = {}
for d in sorted(glob.glob(f"{B}/*/")):
    name = os.path.basename(d.rstrip("/")); R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}ep*.json"))]
    if not R: continue
    E = [e for r in R for e in r["events"] if "last_phase" in e]; EV[name] = E; c = collections.Counter(e["last_phase"] for e in E)
    print(f"| {name} | {len(R)} | {sum(r['ok'] for r in R)} | {len(E)} | " + " | ".join(str(c.get(p, 0)) for p in PH) + " |")
print("\n## 表S2 停止の直前の様子(中央値)")
print("| 方策 | 局面 | 件数 | 掴む位置まで[mm] | 置く位置まで[mm] | 直前100歩の最小(掴む/置く)[mm] | 指の指令の食い違い(直前100歩) | 生徒の動きの大きさ | 先生の動きの大きさ |")
print("|---|---|---|---|---|---|---|---|---|")
for name, E in EV.items():
    for p in PH:
        S = [e for e in E if e["last_phase"] == p]
        if not S: continue
        m = lambda k: np.median([e[k] for e in S])
        print(f"| {name} | {p} | {len(S)} | {m('dg_mm'):.1f} | {m('dp_mm'):.1f} | {m('min_dg_mm'):.1f} / {m('min_dp_mm'):.1f} | {m('grip_disagree_last100'):.0f} | {m('student_move'):.3f} | {m('teacher_move'):.3f} |")
print("\n## 表S3 局面ごとの歩数の割合と、指の指令の食い違い率(全歩)")
print("| 方策 | " + " | ".join(PH) + " |"); print("|---|" + "---|"*len(PH))
for d in sorted(glob.glob(f"{B}/*/")):
    name = os.path.basename(d.rstrip("/")); F = sorted(glob.glob(f"{d}ep*.npz"))
    if not F: continue
    L = np.concatenate([np.load(f)["log"] for f in F]); ph = L[:, 2].astype(int); dis = (L[:, 6] > 0) != (L[:, 7] > 0)
    print(f"| {name} | " + " | ".join(f"{100*np.mean(ph == i):.0f}% ({100*dis[ph == i].mean():.1f}%)" if np.any(ph == i) else "—" for i in range(len(PH))) + " |")
print("\n(表S3の括弧内は、その局面で生徒と先生の指の指令が食い違った歩の割合)")
print("\n## 記録の最後")
print("\n".join(open(f"{B}/stall.log", encoding="utf-8").read().splitlines()[-5:]))
