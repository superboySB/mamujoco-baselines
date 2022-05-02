from functools import partial
from .particle import Particle
from .multiagentenv import MultiAgentEnv
from .mujoco_multi import MujocoMulti


def env_fn(env, **kwargs) -> MultiAgentEnv:  # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["particle"] = partial(env_fn, env=Particle)
REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
