exec > /content/v9.log 2>&1
(cd /content && python3 -u bgm_synth_1005b.py > /content/bgm_v9.log 2>&1 && ffmpeg -y -loglevel error -i /content/full.wav -c:a aac -b:a 192k /content/helicoid_bgm_1005b.m4a && echo BGM_DONE) &
cd /content/pai7 && MUJOCO_GL=egl /content/ev310/bin/python -u build_v9.py ${HELICOID_DRIVE:-drive_out}/perf_seisho/student_1005/s70300 /content/rec /content/seisho_v9_noaudio.mp4
wait
ffmpeg -y -loglevel error -i /content/seisho_v9_noaudio.mp4 -i /content/helicoid_bgm_1005b.m4a -map 0:v -map 1:a -c:v copy -c:a copy -movflags +faststart "-metadata" "title=螺旋体　静かに立ち上がる / Helicoid, Quietly Rising" "-metadata" "artist=Hiroshi Chitose" "-metadata" "copyright=© 2026 Hiroshi Chitose. All rights reserved." "-metadata" "date=2026" /content/seisho_v9.mp4 && echo MUX_DONE
cp /content/seisho_v9.mp4 /content/helicoid_bgm_1005b.m4a ${HELICOID_DRIVE:-drive_out}/perf_seisho/student_1005/ && cp /content/pai7/build_v9.py /content/pai7/opening_type.py /content/bgm_synth_1005b.py ${HELICOID_DRIVE:-drive_out}/code/ && echo SAVED_TO_DRIVE
