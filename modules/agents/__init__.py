REGISTRY = {}

from .comix_agent import CEMAgent, NAFAgent
from .mlp_agent import MLPAgent

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent

REGISTRY["naf"] = NAFAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["mlp"] = MLPAgent