import torch
import torch.nn as nn

from optimbench.tasks import Task


class MLPPINN(nn.Module):
    def __init__(self, width=64, depth=6):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


class BurgersTask(Task):
    name = "burgers_pinn"
    higher_is_better = False
    max_steps_cap = 15000
    checkpoint_every = 150

    def __init__(self, nu=0.01 / 3.14159265, n_collocation=2000, n_boundary=200, n_initial=200):
        self.nu = nu
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial

    def build_model(self, seed):
        torch.manual_seed(seed)
        return MLPPINN()

    def build_data(self, seed):
        return torch.Generator().manual_seed(seed), None

    def sample_batch(self, state):
        return state, state

    def loss_and_metric(self, model, batch):
        generator = batch
        device = self.device
        x_f = ((torch.rand(self.n_collocation, 1, generator=generator) * 2 - 1).to(device)).requires_grad_(True)
        t_f = torch.rand(self.n_collocation, 1, generator=generator).to(device).requires_grad_(True)
        u = model(x_f, t_f)
        u_t = torch.autograd.grad(u, t_f, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_f, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f, torch.ones_like(u_x), create_graph=True)[0]
        residual = u_t + u * u_x - self.nu * u_xx
        f_loss = (residual ** 2).mean()

        x_b = torch.cat([torch.full((self.n_boundary // 2, 1), -1.0), torch.full((self.n_boundary // 2, 1), 1.0)]).to(device)
        t_b = torch.rand(self.n_boundary, 1, generator=generator).to(device)
        u_b = model(x_b, t_b)
        b_loss = (u_b ** 2).mean()

        x_i = (torch.rand(self.n_initial, 1, generator=generator) * 2 - 1).to(device)
        t_i = torch.zeros(self.n_initial, 1, device=device)
        u_i = model(x_i, t_i)
        target_i = -torch.sin(3.14159265 * x_i)
        i_loss = ((u_i - target_i) ** 2).mean()

        return f_loss + b_loss + i_loss

    def evaluate(self, model, val_loader):
        generator = torch.Generator().manual_seed(12345)
        x = (torch.rand(1000, 1, generator=generator) * 2 - 1).to(self.device)
        t = torch.rand(1000, 1, generator=generator).to(self.device)
        with torch.no_grad():
            u = model(x, t)
        return float((u ** 2).mean().item())
