import torch
import torch.nn as nn
from .actor_critic import get_activation


class AdversarialPolicy(nn.Module):
    """对抗策略网络，输出扰动参数"""

    def __init__(self, obs_dim, action_dim, activation, hidden_dims=[256, 256]):
        super().__init__()

        self.activation = get_activation(activation)
        self.softplus = torch.nn.Softplus()

        net = []
        net.append(nn.Linear(obs_dim, hidden_dims[0]))
        net.append(self.activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                net.append(nn.Linear(hidden_dims[l], action_dim))
            else:
                net.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                net.append(self.activation)
        self.net = nn.Sequential(*net)

    def forward(self, obs):
        return self.net(obs)  # 输出扰动参数
