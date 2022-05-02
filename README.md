# mamujoco-baselines

###  1. Install

Install requirements

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

Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).

Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

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

### 2. MAPSA  for MA-mujoco 
#### Results (put tf_events in ./tb_results)



### 3. Some baseline for MA-mujoco 

Train Algorithms (config names): 
- COMIX（comix，comix-naf）
- COVDN（covdn，covdn-naf）
- ~~IQL（iqn-cem，iql-naf）~~
- MADDPG（maddpg）
- FacMADDPG（facmaddpg）
- IPPO（ippo，ippo_ns）
- mappo（mappo，mappo_ns）

``` sh
python main.py --config=mappo --env-config=mujoco_multi with env_args.scenario="HalfCheetah-v2"
```
#### Results (put tf_events in ./tb_results)


##### 2-Agent Ant

```
env_args.scenario="Ant-v2"
env_args.agent_conf="2x4"
env_args.agent_obsk=1
```



##### 2-Agent Ant Diag

```
env_args.scenario="Ant-v2"
env_args.agent_conf="2x4d"
env_args.agent_obsk=1
```

##### 4-Agent Ant

```
env_args.scenario="Ant-v2"
env_args.agent_conf="4x2"
env_args.agent_obsk=1
```

##### 2-Agent HalfCheetah

```
env_args.scenario="HalfCheetah-v2"
env_args.agent_conf="2x3"
env_args.agent_obsk=1
```
result:
![initial results](https://img-blog.csdnimg.cn/de2e13b6b39c47f1acfce1f378be397c.png)

##### 6-Agent HalfCheetah

```
env_args.scenario="HalfCheetah-v2"
env_args.agent_conf="6x1"
env_args.agent_obsk=1
```

##### 3-Agent Hopper

```
env_args.scenario="Hopper-v2"
env_args.agent_conf="3x1"
env_args.agent_obsk=1
```

##### 2-Agent Humanoid

```
env_args.scenario="Humanoid-v2"
env_args.agent_conf="9|8"
env_args.agent_obsk=1
```

##### 2-Agent HumanoidStandup

```
env_args.scenario="HumanoidStandup-v2"
env_args.agent_conf="9|8"
env_args.agent_obsk=1
```

### 2-Agent Reacher

```
env_args.scenario="Reacher-v2"
env_args.agent_conf="2x1"
env_args.agent_obsk=1
```

### 2-Agent Swimmer

```
env_args.scenario="Swimmer-v2"
env_args.agent_conf="2x1"
env_args.agent_obsk=1
```

### 2-Agent Walker

```
env_args.scenario="Walker2d-v2"
env_args.agent_conf="2x3"
env_args.agent_obsk=1
```

### Manyagent Swimmer

```
env_args.scenario="manyagent_swimmer"
env_args.agent_conf="10x2"
env_args.agent_obsk=1
```

### Manyagent Ant

```
env_args.scenario="manyagent_ant"
env_args.agent_conf="2x3"
env_args.agent_obsk=1
```

### Coupled HalfCheetah (NEW!)

```
env_args.scenario="coupled_half_cheetah"
env_args.agent_conf="1p1"
env_args.agent_obsk=1
```

`CoupledHalfCheetah` features two separate HalfCheetah agents coupled by an elastic tendon. You can add more tendons or novel coupled scenarios by
