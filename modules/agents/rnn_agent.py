import torch.nn as nn
import torch.nn.functional as F
from utils.distributions import DiagGaussian


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.action_out = DiagGaussian(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        if self.args.use_rnn:
            h_in = hidden_state.reshape(-1, self.args.hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        h_out = self.fc2(h)

        action_logits = self.action_out(h_out)
        # action = action_logits.sample()
        action = F.tanh(action_logits.sample())
        action_log_prob = action_logits.log_probs(action)
        dist_entropy= action_logits.entropy()

        return action, action_log_prob, dist_entropy, h

