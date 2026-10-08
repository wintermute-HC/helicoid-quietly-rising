# 螺旋体　静かに立ち上がる / Helicoid, Quietly Rising

一行の言葉から、Claude が 5 段の石の「回転の並び」を構成し、模倣学習で育てた方策がロボットアームを動かして、円環の外から石を運び、円環の中に静かに積み上げる作品です。

1. **言葉** — 例:「静かに立ち上がる螺旋」
2. **構成** — Claude が各段の回転角を提案し、完成予想図を描いて自分で見て、修正を重ねて 5 段の構成を確定する
3. **組み上げ** — 学習した方策(生徒)が、シミュレーション上のロボット(Panda)で円環の外の石を一つずつ掴み、円環を越えて運び、構成どおりに積む
4. **映像** — 構成の過程と組み上げを撮影し、字幕・音楽と合わせて一本の映像にする

## 構成

```
.
├── sim/        環境・知覚・制御
│   ├── tower_env.py      円環と台座のある積み石環境(robosuite / MuJoCo)
│   ├── ring_env.py       円環を越える運搬の環境
│   ├── twist_env.py      回転つき積み石の基本環境と幾何ユーティリティ
│   ├── geo.py            幾何計算
│   ├── perception.py     上方カメラ画像からの石・円環の検出
│   ├── expert.py         ピック&プレースのスクリプト制御(基礎)
│   ├── ring_expert.py    円環を越える運搬高さを持つスクリプト制御
│   ├── seeing_expert.py  知覚を使って石を探し積む制御(先生)
│   ├── tower_expert.py   塔の組み上げ制御と出来の判定
│   └── verify.py         構成案の高速検査(倒れにくさ・物理での自立確認)
├── learning/   模倣学習(行動クローン + DAgger)
│   ├── markov_teacher.py  状態だけから行動を決める先生
│   ├── learn_common.py    特徴量・小型方策(NumPy 推論)など共通部品
│   ├── test_markov.py     先生/生徒を切り替えて塔を組む実行器
│   ├── mk_worker.py       教材集め・評価を並列で回すワーカー
│   ├── train_bc.py        行動クローンの学習(PyTorch)
│   ├── dagger.py          DAgger の 1 ラウンド
│   ├── pipeline3.py       DAgger の教材集め(r0〜r5)と、方策 pol_r0〜pol_r4 の学習
│   ├── eval_r4.py         pol_r4 の評価
│   ├── r5.py              最終方策 pol_r5 の学習と評価
│   └── diag_1004.py       失敗場面の診断
├── compose/    構成AI
│   ├── composer.py   言葉 → Claude による構成と自己修正のループ
│   ├── perform.py    構成から撮影までの一括実行
│   ├── preview.py    完成予想図の描画
│   └── run02/        映像に使った構成記録(run02.json と各回の予想図 iter01〜05.png)
├── film/       撮影と映像
│   ├── film2.py             組み上げの撮影(先生/生徒を切替)
│   ├── build_seisho.py      字幕・構成記録・撮影素材から映像を組む
│   ├── build_v9.py          最終版の映像組み立て
│   ├── opening_type.py      冒頭の文字の打ち出し
│   ├── screens.json         字幕の文章
│   └── bgm_synth_1005c.py   音楽の合成
├── policy/pol_r5.npz   最終方策(生徒)
└── archive/    試作・旧版(参考として残しているもの。セットアップ用の setup_env.sh もここ)
```

各フォルダのスクリプトは、冒頭で `sim/ learning/ compose/ film/ archive/` を import パスに加えます(動作確認は構文チェックのみ)。

## 結果

- 最終方策 `policy/pol_r5.npz` は、先生の助けなし(生徒単独)で、学習に使っていない **未知の 44 場面のうち 43 場面で 5 段の塔を組み上げました**(`learning/r5.py` による評価)。


## 評価場面のseed一覧 / Evaluation seeds

最終方策 `pol_r5` の評価に使った44場面です（seed 70000〜71003）。どの場面も学習には使っていません。学習データのseedは 0〜51001 と 60000〜61003 で、重なりはありません。半数の場面は乱れありで、先生（制御器）は介入しません。結果は 43/44（乱れなし 21/22・乱れあり 22/22）でした。
The 44 scenes (seeds 70000–71003) used to evaluate the final policy `pol_r5`. None was used in training (training seeds: 0–51001 and 60000–61003). Half of the scenes are perturbed, and the teacher never intervenes. Result: 43/44 (clean 21/22, perturbed 22/22).

- 一つ前の方策 pol_r4 は、別の44場面（seed 60000〜61003）で 38/44 でした。その評価で集めた添削を学習に加えたものが pol_r5 です。
  The previous policy pol_r4 scored 38/44 on a separate set (seeds 60000–61003); its corrections were added to train pol_r5.
- いずれも同じ環境・課題での評価です（石5個、円環1基、同じカメラと物理条件）。
  All scenes share the same environment and task (5 stones, 1 ring, same camera and physics).

○ 成功 / success　× 失敗 / failure（1個目を掴む段階で停止 / stalled at the first grasp）　🎬 映像に使用 / used in the film

| | | | |
|---|---|---|---|
| 70000 ○ | 70001 ○ | 70002 × | 70003 ○ |
| 70100 ○ | 70101 ○ | 70102 ○ | 70103 ○ |
| 70200 ○ | 70201 ○ | 70202 ○ | 70203 ○ |
| 70300 ○ 🎬 | 70301 ○ | 70302 ○ | 70303 ○ |
| 70400 ○ | 70401 ○ | 70402 ○ | 70403 ○ |
| 70500 ○ | 70501 ○ | 70502 ○ | 70503 ○ |
| 70600 ○ | 70601 ○ | 70602 ○ | 70603 ○ |
| 70700 ○ | 70701 ○ | 70702 ○ | 70703 ○ |
| 70800 ○ | 70801 ○ | 70802 ○ | 70803 ○ |
| 70900 ○ | 70901 ○ | 70902 ○ | 70903 ○ |
| 71000 ○ | 71001 ○ | 71002 ○ | 71003 ○ |


## 再現手順

動作確認環境: **Google Colab(GPU: L4)**、**Python 3.10**、`mujoco==2.3.7`、`robosuite==1.4.1`、`numpy<2`。

```bash
# 1. 取得
git clone <this repository> /content/helicoid && cd /content/helicoid

# 2. Python 3.10 の仮想環境(/content/ev310)を作り、依存を入れる
#    (中身: numpy<2, mujoco==2.3.7, robosuite==1.4.1, pillow, scipy)
bash archive/setup_env.sh && tail -n 2 /content/setup_env.log

# 3. 成果の保存先(任意)。Google Drive に残す場合はマウントしたフォルダを指定
export HELICOID_DRIVE=/content/drive/MyDrive/<任意のフォルダ>   # 既定: ./drive_out
export MUJOCO_GL=egl

# 4. 模倣学習(作業領域は /content/mk7。train_bc.py は torch の入った python3 で実行される)
python3 learning/pipeline3.py      # DAgger の教材集め(r0〜r5)と、方策 pol_r0〜pol_r4 の学習
python3 learning/eval_r4.py        # pol_r4 の評価(この結果も r5 の教材になる)
python3 learning/r5.py             # 最終方策 pol_r5 の学習と未知場面での評価

# 5. 生徒単独での撮影(学習を省く場合は同梱の policy/pol_r5.npz を使う)
#    例の REL・SEED は映像に使った試行
#    構成記録の最終案(run02.json)は Claude が出力した整数の並び 0,22,23,22,23(累積90°)。
#    映像の撮影では、同じ累積90°を 22.5° ずつ等分して実行した
ACTOR=student POL=policy/pol_r5.npz REL=0,22.5,22.5,22.5,22.5 SEED=70300 OUT=out/film \
  /content/ev310/bin/python film/film2.py

# 6. 構成(Claude API を使う。ANTHROPIC_API_KEY を環境変数で与え、anthropic を pip で入れる)
/content/ev310/bin/python compose/perform.py --words "静かに立ち上がる螺旋" --out out/perf
#    注: perform.py の通し実行は旧版の撮影(archive/film.py、先生の制御)を使う。生徒による撮影は手順5

# 7. 映像の組み立て(ffmpeg、Noto Serif CJK・Lora・M PLUS 1 Code・JetBrains Mono の各書体が必要)
#    構成記録フォルダには、映像に使った構成記録 compose/run02 を使う
/content/ev310/bin/python film/build_v9.py <撮影素材フォルダ> compose/run02 out/helicoid.mp4
python3 film/bgm_synth_1005c.py    # 音楽(full.wav)の合成
```

注意:
- API キーなどの秘密情報はリポジトリに含めていません。Claude API を使う場合は `ANTHROPIC_API_KEY` を各自の環境変数で設定してください。
- 書体のパス(`/usr/share/fonts/...`、`/content/fonts_mono`)は Colab での配置を前提にしています。

## 権利表示

- **コード**: MIT License(`LICENSE` を参照)。Copyright (c) 2026 Chitose10
- **映像・字幕の文章(`film/screens.json` ほか)・音楽・構成記録(`compose/run02/`)**: © 2026 Chitose10. All rights reserved.

---

## Summary (English)

*Helicoid, Quietly Rising* turns a single line of words into a sculpture. Claude composes the rotations of a five-stone tower, checking its own rendered previews and revising them; a policy trained by imitation learning (behavior cloning + DAgger) then drives a simulated Panda arm to carry each stone from outside a ring and stack it inside. Acting alone, the final student policy (`policy/pol_r5.npz`) completed the tower in 43 of 44 unseen scenes. Developed and run on Google Colab (L4) with Python 3.10, mujoco 2.3.7, robosuite 1.4.1 and numpy<2. Code is MIT-licensed; the film, subtitle text, music and composition record (compose/run02/) are © 2026 Chitose10. All rights reserved.
