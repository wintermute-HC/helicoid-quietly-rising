exec > /content/bgm_1005.log 2>&1
cd /content && python3 -u bgm_synth_1005.py && echo SYNTH_DONE
ffmpeg -y -loglevel error -i /content/full.wav -c:a aac -b:a 192k /content/helicoid_bgm_1005.m4a
ffmpeg -y -loglevel error -i /content/seisho_v8_noaudio.mp4 -i /content/helicoid_bgm_1005.m4a -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k /content/seisho_v8_preview.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 /content/seisho_v8_preview.mp4
cp /content/helicoid_bgm_1005.m4a /content/seisho_v8_preview.mp4 ${HELICOID_DRIVE:-drive_out}/perf_seisho/student_1005/ && cp /content/bgm_synth_1005.py ${HELICOID_DRIVE:-drive_out}/code/ && echo SAVED_TO_DRIVE
