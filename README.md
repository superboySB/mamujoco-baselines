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
# if x86_64-linux-gnu failed in Ubuntu 16.04, run "sudo apt-get install libglew-dev build-essential libssl-dev libffi-dev python3-dev libosmesa6-dev patchelf"
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
# Q: if raise error "libffi.so.7: cannot open shared object file"
# A: cd your_conda_path/mujoco/lib && ln -s libffi.so.6 libffi.so.7
# Q: if raise errorfrom LD_PRELOAD cannot be preloaded
# A: unset LD_PRELOAD (do not set LD_PRELOAD)
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

替换算法需要将comix/covdn/maddpg/facmaddpg/ippo/mappo这几个key替换到下面config的指定位置。替换环境需要将下面Results目录下各个环境的相应参数替换到env_args的对应key上，不同环境的tensorbaord结果可以保存在tb_results的对应文件夹中。例如：

``` sh
python main.py --config=comix --env-config=mujoco_multi with env_args.scenario="Ant-v2" env_args.agent_conf="2x4" env_args.agent_obsk=1
```
#### Results (put tf_events in ./tb_results)
##### 2-Agent Ant

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="Ant-v2" env_args.agent_conf="2x4" env_args.agent_obsk=1
```
result:

##### 2-Agent Ant Diag

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="Ant-v2" env_args.agent_conf="2x4d" env_args.agent_obsk=1
```
result:
##### 4-Agent Ant

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="Ant-v2" env_args.agent_conf="4x2" env_args.agent_obsk=1
```
result:
##### 2-Agent HalfCheetah

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="HalfCheetah-v2" env_args.agent_conf="2x3" env_args.agent_obsk=1
```
result:
![trial results](https://img-blog.csdnimg.cn/1e21756925314eaebf321536678b81f2.png)

##### 6-Agent HalfCheetah

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="HalfCheetah-v2" env_args.agent_conf="6x1" env_args.agent_obsk=1
```
result:
##### 3-Agent Hopper

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="Hopper-v2" env_args.agent_conf="3x1" env_args.agent_obsk=1
```
result:
##### 2-Agent Humanoid

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="Humanoid-v2" env_args.agent_conf="9|8" env_args.agent_obsk=1
```
result:
##### 2-Agent HumanoidStandup

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="HumanoidStandup-v2" env_args.agent_conf="9|8" env_args.agent_obsk=1
```
result:
##### 2-Agent Reacher

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="Reacher-v2" env_args.agent_conf="2x1" env_args.agent_obsk=1
```
result:
##### 2-Agent Swimmer

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="Swimmer-v2" env_args.agent_conf="2x1" env_args.agent_obsk=1
```
result:
##### 2-Agent Walker

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="Walker2d-v2" env_args.agent_conf="2x3" env_args.agent_obsk=1
```
result:
##### Manyagent Swimmer

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="manyagent_swimmer" env_args.agent_conf="10x2" env_args.agent_obsk=1
```
result:
##### Manyagent Ant

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="manyagent_ant" env_args.agent_conf="2x3" env_args.agent_obsk=1
```
result:
##### Coupled HalfCheetah (NEW!)

```sh
python3.7 main.py --config=alg_name --env-config=mujoco_multi with env_args.scenario="coupled_half_cheetah" env_args.agent_conf="1p1" env_args.agent_obsk=1
```
result:
## Reference
* https://github.com/schroederdewitt/multiagent_mujoco
* https://github.com/oxwhirl/comix
