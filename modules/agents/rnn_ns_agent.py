import torch.nn as nn
from modules.agents.rnn_agent import RNNAgent
import torch as th


class RNNNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNNSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        actions = []
        action_log_probs = []
        es = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                action, action_log_prob, e, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                actions.append(action)
                action_log_probs.append(action_log_prob)
                es.append(e)

            return th.cat(actions), th.cat(action_log_probs), th.cat(es), th.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                if hidden_state is not None:
                    action, action_log_prob, e, h = self.agents[i](inputs[:, i], hidden_state[:, i])
                else:
                    action, action_log_prob, e, h = self.agents[i](inputs[:, i], None)
                hiddens.append(h.unsqueeze(1))
                actions.append(action.unsqueeze(1))
                action_log_probs.append(action_log_prob.unsqueeze(1))
                es.append(e)
            return th.cat(actions, dim=-1).view(-1, action.size(-1)), \
                   th.cat(action_log_probs, dim=-1).view(-1, action_log_prob.size(-1)), \
                   th.cat(es, dim=-1).view(-1, e.size(-1)), \
                   th.cat(hiddens, dim=1)

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
