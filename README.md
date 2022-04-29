# mamujoco-baselines

###  1. Install

1. Install requirements
```shell
apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
conda create -n mujoco python==3.8
conda activate mujoco
conda install pytorch torchvision torchaudio cudatoolkit=11.3 tensorboard future
```

2. Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
3. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

```sh
# add to ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liuchi/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# install the pip dependencies
pip3 install -U 'mujoco-py<2.2,>=2.1'
pip install -r requirements.txt
python render_mujoco.py  # test mujoco
python render_mamujoco.py  # test mujoco_multi
```

### 2.1 MAPSA  for MA-mujoco 
Results



### 2.2 Some baseline for MA-mujoco 

Train: 
- COMIX（comix，comix-naf）
- COVDN（covdn，covdn-naf）
- IQL（iqn-cem，iql-naf）*cannot run*
- MADDPG（maddpg）
- FacMADDPG（facmaddpg）

``` sh
python main.py --config=comix --env-config=mujoco_multi with env_args.scenario="HalfCheetah-v2"
```
Results
