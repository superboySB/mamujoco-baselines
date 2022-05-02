from .cq_learner import CQLearner
from .facmaddpg_learner import FacMADDPGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner

REGISTRY = {}

REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmaddpg_learner"] = FacMADDPGLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner