# from https://github.com/whitneychiu/lipmlp_pytorch/
# modified by: @Zhang Yf

import torch
import math


class LipschitzLinear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max()  # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)


class lipmlp(torch.nn.Module):

    def __init__(self, dims):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims) - 2):
            self.layers.append(LipschitzLinear(dims[ii], dims[ii + 1]))

        self.layer_output = LipschitzLinear(dims[-2], dims[-1])
        self.relu = torch.nn.ReLU()

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        loss_lipc = loss_lipc * self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        return self.layer_output(x)


# example code, not runnable
if __name__ == "__main__":
    resolution = 32
    V = sample_2D_grid(resolution)  # |V|x2
    gt0 = sdf_cross(V)
    gt1 = sdf_star(V)
    latent0 = torch.tensor([0])
    latent1 = torch.tensor([1])

    dims = [3, 64, 64, 64, 64, 1]
    model = lipmlp(dims)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 200000
    pbar = tqdm(range(num_epochs))

    V0 = torch.hstack((V, latent0.repeat(V.shape[0], 1))).to(device)  # nVx3
    V1 = torch.hstack((V, latent1.repeat(V.shape[0], 1))).to(device)  # nVx3
    gt0 = gt0.to(device)
    gt1 = gt1.to(device)
    model = model.to(device)

    loss_history = []
    loss_sdf_history = []
    loss_lipschitz_history = []
    lam = 1e-5
    for epoch in pbar:
        # forward
        sdf0 = model(V0).squeeze(1)
        sdf1 = model(V1).squeeze(1)

        # compute loss
        loss_sdf = loss_func(sdf0, gt0) + loss_func(sdf1, gt1)
        loss_lipschitz = lam * model.get_lipschitz_loss()
        loss = loss_sdf + loss_lipschitz

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        loss_sdf_history.append(loss_sdf.item())
        loss_lipschitz_history.append(loss_lipschitz.item())
