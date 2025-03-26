#Desc: 用于测试模型的计算量和参数量
import torch
import torch.nn as nn
from thop import profile

# 定义一个简单的模型
class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        mlp_input_dim = 400
        num_actions = 10
        actor_hidden_dims=[512, 256, 256, 128]
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim, actor_hidden_dims[0]))
        actor_layers.append(nn.ReLU())
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_layers)


    def forward(self, x):
        return self.actor(x)


model = SimpleMLP()
input_tensor = torch.randn(1, 400)

# 使用 thop 计算 FLOPs 和参数量
flops, params = profile(model, inputs=(input_tensor,))
print(f"Total FLOPs: {flops}")
print(f"Total Params: {params}")      