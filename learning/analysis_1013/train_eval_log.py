# 切り分け実験(10/12)の学習と評価の記録の集計(教材の本数・サンプル数・検証損失・所要時間)。論文 付録Cの表C-1。
import glob, json, os, re, datetime
B = os.path.join(os.environ.get("HELICOID_DRIVE", "drive_out"), "bc_ablate_1012")
print("JST", datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%m/%d %H:%M"))
print("## 学習の記録(教材の本数・サンプル数・最終損失)")
for f in sorted(glob.glob(f"{B}/train_*.log")):
    L = open(f, encoding="utf-8", errors="ignore").read().splitlines()
    ep = next((l for l in L if l.startswith("episodes")), "")
    sv = next((l for l in L if l.startswith("saved")), "")
    last = next((l for l in reversed(L) if l.startswith("{")), "")
    print(os.path.basename(f), "|", ep, "|", re.sub(r"^saved \S+ ", "", sv), "|", last)
print("\n## 評価の所要時間(1場面あたりの秒数の合計と平均)")
for d in sorted(glob.glob(f"{B}/eval/*")):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/ep*.json"))]
    if R: s = sum(r["sec"] for r in R); print(f"{os.path.basename(d)}: {len(R)}場面, 合計{s/60:.0f}分, 平均{s/len(R):.0f}秒")
print("\n## 全体の記録(ablate.log の最初と最後)")
L = open(f"{B}/ablate.log", encoding="utf-8", errors="ignore").read().splitlines()
print("\n".join(L[:3])); print("..."); print("\n".join(L[-3:]))
