# 言葉 → Claudeが5段の「回転の並び」を提案 → 完成予想図を描いてClaude自身に見せる → Claudeが見て修正、を繰り返す
# (塔の軸は垂直に保つ。はみ出し・傾きは作品の美意識に合わないため使わない: 9/29 作家の判断)
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # フォルダ分け(sim/learning/compose/film/archive)に伴うimportパス
_sys.path[1:1] = [_p for _p in (_os.path.join(_R, _d) for _d in ("sim", "learning", "compose", "film", "archive")) if _p not in _sys.path]
import os, re, json
from verify import check
from preview import render, png_b64
MODEL = os.environ.get("COMPOSER_MODEL", "claude-opus-5-5")
SYSTEM = """あなたは彫刻家の協働者として、ロボットが積む5段の大理石の塔の「構成」を設計します。
作品: 台座の代わりに白大理石の円環を地に置き、その中央に一辺50mmの立方体を5個、垂直に積む。色は下から 黒・白・黒・白・黒 で固定。
ロボットは円環の外にある石を1個ずつ運び込み、あなたの構成どおりに積む。塔の軸は常に垂直で、石をずらしたり傾けたりはしない。
あなたが決めるのは各段の回転だけ:
  rel_deg: 1段目は主カメラに対する向き(度、-45〜45)、2段目以降は「すぐ下の段に対する回転角」(-45〜45)
  立方体は90°ごとに同じ形になるので、±45°が最大のねじれ。0°は下の段と面がそろう。
作品の思想: 円は何も閉ざさず、内と外を区別するだけ。塔はわずかずつ向きを変えながら立ち上がり、自らと環境との差異を刻む。
方針: 与えられた言葉の意味と情感を、回転の並び(リズム・加速・反転・静止)に翻訳すること。
提案のたびに、その構成の完成予想図が2枚返ってくる(左: 作品を撮る主カメラ=斜め上から、右: 低い横から)。
必ず画像を見て、言葉が形として立ち現れているかを自分で批評し、足りなければ修正すること。見て納得できたら "final": true を付ける。
出力は次のJSONだけを ```json ``` で囲んで返す:
{"title": "短い作品名", "concept": "言葉をどう回転の並びに翻訳したか(2〜3文)", "critique": "前の予想図を見ての自己批評(初回は空でよい)",
 "levels": [{"rel_deg": 0}, {"rel_deg": 18}, ... 5段], "final": false}"""
def _parse(text):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    d = json.loads(m.group(1)); lv = d["levels"]; assert len(lv) == 5, "5段ではない"
    d["levels"] = [dict(color=c, rel_deg=max(-45.0, min(45.0, float(l["rel_deg"]))), dx_mm=0.0, dy_mm=0.0) for l, c in zip(lv, "BWBWB")]
    return d
def claude_propose(messages):
    import anthropic
    r = anthropic.Anthropic().messages.create(model=MODEL, max_tokens=8000, system=SYSTEM, messages=messages)
    return "".join(b.text for b in r.content if b.type == "text")      # 思考ブロックを除き本文だけ取り出す
def compose(words, propose=claude_propose, max_iter=5, log_path=None, img_dir=None):
    msgs = [{"role": "user", "content": f"言葉: 「{words}」\nこの言葉から5段の回転の並びを提案してください。"}]; trace = []
    for it in range(1, max_iter + 1):
        text = propose(msgs); msgs.append({"role": "assistant", "content": text})
        try: d = _parse(text)
        except Exception as e:
            msgs.append({"role": "user", "content": f"JSONを読み取れませんでした({e})。形式どおりに出し直してください。"}); continue
        v = check(d["levels"], 0.0)                       # 垂直積みなので安全確認のみ(円環との干渉・倒れないこと)
        img = render(d["levels"])
        if img_dir:
            from PIL import Image; os.makedirs(img_dir, exist_ok=True); Image.fromarray(img).save(f"{img_dir}/iter{it:02d}.png")
        trace.append(dict(iter=it, title=d.get("title"), concept=d.get("concept"), critique=d.get("critique"),
                          rel_deg=[l["rel_deg"] for l in d["levels"]], final=d.get("final", False), physics_ok=v["ok"]))
        print(f"[{it}] {d.get('title')} | 回転 {[round(l['rel_deg']) for l in d['levels']]} | 物理{'OK' if v['ok'] else 'NG'} | final={d.get('final', False)}", flush=True)
        if d.get("critique"): print("    自己批評:", d["critique"], flush=True)
        if log_path: json.dump(dict(words=words, model=MODEL, trace=trace), open(log_path, "w"), ensure_ascii=False, indent=1)
        if d.get("final") and v["ok"]: return d, trace
        msgs.append({"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64(img)}},
            {"type": "text", "text": ("この構成の完成予想図です(左: 主カメラ、右: 低い横から)。" if v["ok"] else "物理的に立たない構成でした。") +
             "言葉と照らして批評し、修正案を出すか、納得できれば同じ構成に final: true を付けて返してください。"}]})
    return (trace and dict(levels=[dict(color=c, rel_deg=r, dx_mm=0.0, dy_mm=0.0) for c, r in zip("BWBWB", trace[-1]["rel_deg"])],
                           title=trace[-1]["title"], concept=trace[-1]["concept"])), trace
