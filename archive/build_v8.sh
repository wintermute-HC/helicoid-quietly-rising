exec > /content/build_v8.log 2>&1
apt-get install -y -qq fonts-noto-cjk fonts-noto-cjk-extra
mkdir -p /usr/share/fonts/truetype/google-fonts
curl -sL -o /usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf "https://github.com/google/fonts/raw/main/ofl/lora/Lora%5Bwght%5D.ttf"
curl -sL -o /usr/share/fonts/truetype/google-fonts/Lora-Italic-Variable.ttf "https://github.com/google/fonts/raw/main/ofl/lora/Lora-Italic%5Bwght%5D.ttf"
ls -l /usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc /usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc /usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc /usr/share/fonts/truetype/google-fonts/Lora-*.ttf
echo FONTS_DONE
cd /content/pai7 && MUJOCO_GL=egl /content/ev310/bin/python -u build_v8.py ${HELICOID_DRIVE:-drive_out}/perf_seisho/student_1005/s70300 /content/rec /content/seisho_v8_noaudio.mp4
cp /content/seisho_v8_noaudio.mp4 /content/build_v8.log ${HELICOID_DRIVE:-drive_out}/perf_seisho/student_1005/ && echo SAVED_TO_DRIVE
