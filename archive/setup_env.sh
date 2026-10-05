set -e
exec > /content/setup_env.log 2>&1
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
python3.10 -m venv /content/ev310
/content/ev310/bin/pip install -q "numpy<2" mujoco==2.3.7 robosuite==1.4.1 pillow scipy
MUJOCO_GL=egl /content/ev310/bin/python -c "import numpy,mujoco,robosuite;print('numpy',numpy.__version__,'mujoco',mujoco.__version__,'robosuite',robosuite.__version__)"
echo SETUP_DONE
