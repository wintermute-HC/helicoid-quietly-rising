# 切り分け実験(10/12)の詳細集計: 成功率と95%信頼区間(Wilson)、種ごとの結果、段ごとの合格率、成功時の置き精度。論文 4.4節の表A〜C。
# 使い方: HELICOID_DRIVE=<Driveのフォルダ> python3 ablate_detail.py
# セル5(詳細集計) | 時刻：10/13 09:05 JST 作成 | 実行環境：Colab CPU(GPU不要)
import glob, json, os, math, re, collections, numpy as np
import os; B = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "bc_ablate_1012"); E = f"{B}/eval"
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(100*(c-h), 1), round(100*(c+h), 1))
def ok_lv(r): return abs(r["dz_mm"]) < 5 and r["xy_mm"] < 10 and abs(r["yaw_err"]) < 5
runs = {os.path.basename(d): [json.load(open(f)) for f in sorted(glob.glob(f"{d}/ep*.json"))] for d in sorted(glob.glob(f"{E}/*"))}
cond = collections.defaultdict(list)
for name, R in runs.items(): cond[re.sub(r"_s\d+$", "", name)].append((name, R))
out = {}
print("## 表A 条件別(3種まとめ, 95%信頼区間=Wilson)")
print("| 条件 | 成功/計 | 成功率 | 95%CI | 乱れなし | 乱れあり | 成功時の平均手数 | 失敗時の平均手数 |")
print("|---|---|---|---|---|---|---|---|")
for c, L in cond.items():
    R = [r for _, x in L for r in x]; k = sum(r["ok"] for r in R); n = len(R)
    cl = [r for r in R if r["seed"] % 2 == 0]; no = [r for r in R if r["seed"] % 2]
    st_ok = [r["steps"] for r in R if r["ok"]]; st_ng = [r["steps"] for r in R if not r["ok"]]
    lo, hi = wilson(k, n)
    row = dict(k=k, n=n, ci=(lo, hi), clean=(sum(r["ok"] for r in cl), len(cl)), noisy=(sum(r["ok"] for r in no), len(no)),
               steps_ok=round(np.mean(st_ok)) if st_ok else None, steps_ng=round(np.mean(st_ng)) if st_ng else None)
    out[c] = row
    print(f"| {c} | {k}/{n} | {100*k/n:.1f}% | {lo}–{hi}% | {row['clean'][0]}/{row['clean'][1]} | {row['noisy'][0]}/{row['noisy'][1]} | {row['steps_ok']} | {row['steps_ng']} |")
print("\n## 表B 種ごと(付録用)")
print("| 実行 | 成功/計 | 乱れなし | 乱れあり |"); print("|---|---|---|---|")
for name, R in runs.items():
    cl = [r for r in R if r["seed"] % 2 == 0]; no = [r for r in R if r["seed"] % 2]
    print(f"| {name} | {sum(r['ok'] for r in R)}/{len(R)} | {sum(r['ok'] for r in cl)}/{len(cl)} | {sum(r['ok'] for r in no)}/{len(no)} |")
print("\n## 表C 段ごとの合格率(全エピソード, 段の基準: dz±5mm, xy10mm未満, yaw±5°)")
LV = max(len(r["levels"]) for R in runs.values() for r in R)
print("| 条件 | " + " | ".join(f"{i+1}段" for i in range(LV)) + " |"); print("|---|" + "---|"*LV)
for c, L in cond.items():
    R = [r for _, x in L for r in x]
    print(f"| {c} | " + " | ".join(f"{100*np.mean([ok_lv(r['levels'][i]) for r in R if len(r['levels'])>i]):.0f}%" for i in range(LV)) + " |")
print("\n## 表D 成功エピソードの置き精度(段ごと, 中央値 / 95パーセンタイル)")
print("| 条件 | 段 | xy[mm] | dz絶対値[mm] | yaw絶対値[°] | 件数 |"); print("|---|---|---|---|---|---|")
for c, L in cond.items():
    R = [r for _, x in L for r in x if r["ok"]]
    if len(R) < 5: continue
    for i in range(LV):
        v = [r["levels"][i] for r in R]
        f = lambda key, a=False: f"{np.median([abs(x[key]) if a else x[key] for x in v]):.2f} / {np.percentile([abs(x[key]) if a else x[key] for x in v], 95):.2f}"
        print(f"| {c} | {i+1} | {f('xy_mm')} | {f('dz_mm', True)} | {f('yaw_err', True)} | {len(v)} |")
json.dump(out, open(f"{B}/detail_1013.json", "w"), indent=1, ensure_ascii=False); print("\nSAVED", f"{B}/detail_1013.json")
