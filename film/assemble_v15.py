# v15(10/20): 講座運営から配布されたオープニング(OP、約3秒)とエンドタイトル(ED、約3秒)を、提出動画の前後に付ける。
#   要項の「3分以内」に収めるため、本編の最後の1.5秒(著作権画面から無地へのフェードと無地の静止)を外し、
#   OP→本編、本編(著作権画面)→ED を各0.5秒のクロスフェードでつなぐ。OP・EDは無音なので、音楽は本編の区間だけに置き、
#   外した1.5秒の代わりに本編の最後1.5秒で音を消していく。OP・EDは本編に合わせて2560×1440・20fpsに変換する。
#   使い方: python assemble_v15.py <OP.mp4> <本編v14.mp4> <ED.mp4> <出力mp4> [エンコード速度(既定 slow)]
import sys, json, subprocess
OP, MAIN, ED, OUT = sys.argv[1:5]; PRESET = sys.argv[5] if len(sys.argv) > 5 else "slow"
W, H, FPS, X, CUT = 2560, 1440, 20, 0.5, 1.5        # 解像度・fps・クロスフェード秒・本編の末尾から外す秒
dur = lambda p: float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", p],
                                                capture_output=True, text=True).stdout)["format"]["duration"])
LO, LM, LE = 3.0, round(dur(MAIN) - CUT, 2), 2.95  # OP・EDは端の静止部分を少し切って長さをそろえる
T = LO + LM + LE - 2*X; AD = LO - X                 # 全長 / 音楽の開始時刻
print(f"本編 {LM:.2f}秒(元 {dur(MAIN):.2f}秒から末尾{CUT}秒を外す) | 全長 {T:.2f}秒 | 3分以内: {'OK' if T <= 180 else 'NG'}")
norm = f"scale={W}:{H}:flags=lanczos,setsar=1,fps={FPS},format=yuv420p,settb=1/{FPS}"
fc = (f"[0:v]{norm},trim=duration={LO},setpts=PTS-STARTPTS[op];"
      f"[1:v]setsar=1,fps={FPS},format=yuv420p,settb=1/{FPS},trim=duration={LM},setpts=PTS-STARTPTS[m];"
      f"[2:v]{norm},trim=duration={LE},setpts=PTS-STARTPTS[ed];"
      f"[op][m]xfade=transition=fade:duration={X}:offset={LO - X}[a];"
      f"[a][ed]xfade=transition=fade:duration={X}:offset={LO + LM - 2*X},format=yuv420p[v];"
      f"[1:a]atrim=duration={LM},asetpts=PTS-STARTPTS,afade=t=out:st={LM - CUT}:d={CUT},"
      f"adelay={int(AD*1000)}:all=1,apad,atrim=duration={T}[au]")
cmd = ["ffmpeg", "-y", "-v", "error", "-stats", "-i", OP, "-i", MAIN, "-i", ED, "-filter_complex", fc, "-map", "[v]", "-map", "[au]",
       "-r", str(FPS), "-c:v", "libx264", "-preset", PRESET, "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
       "-map_metadata", "-1", "-metadata", "title=螺旋体　静かに立ち上がる / Helicoid, Quietly Rising",
       "-metadata", "copyright=All rights reserved.", "-metadata", "date=2026", "-movflags", "+faststart", OUT]
subprocess.run(cmd, check=True); print("DONE_V15", f"{dur(OUT):.2f}秒")
