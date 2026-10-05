exec > /content/v10.log 2>&1
(cd /content && python3 -u bgm_synth_1005c.py > /content/bgm_v10.log 2>&1 && ffmpeg -y -loglevel error -i /content/full.wav -c:a aac -b:a 192k /content/helicoid_bgm_1005c.m4a && echo BGM_DONE) &
cd /content/pai7 && MUJOCO_GL=egl /content/ev310/bin/python -u build_v9.py ${HELICOID_DRIVE:-drive_out}/perf_seisho/student_1005/s70300 /content/rec /content/seisho_v10_noaudio.mp4
wait
ffmpeg -y -loglevel error -i /content/seisho_v10_noaudio.mp4 -i /content/helicoid_bgm_1005c.m4a -map 0:v -map 1:a -c:v copy -c:a copy -movflags +faststart "-metadata" "title=螺旋体　静かに立ち上がる / Helicoid, Quietly Rising" "-metadata" "artist=Hiroshi Chitose" "-metadata" "copyright=© 2026 Hiroshi Chitose. All rights reserved." "-metadata" "date=2026" /content/seisho_v10.mp4 && echo MUX_DONE
cp /content/seisho_v10.mp4 /content/helicoid_bgm_1005c.m4a ${HELICOID_DRIVE:-drive_out}/perf_seisho/student_1005/ && cp /content/pai7/screens.json ${HELICOID_DRIVE:-drive_out}/code/screens_1005.json && cp /content/bgm_synth_1005c.py ${HELICOID_DRIVE:-drive_out}/code/ && echo SAVED_TO_DRIVE
