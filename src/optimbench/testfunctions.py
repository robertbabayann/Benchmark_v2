import math
import torch
import torch.nn as nn


class TestFunctionModel(nn.Module):
    def __init__(self, function, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.function = function
        self.x = nn.Parameter(torch.empty(function.dim).uniform_(*function.bounds))

    def forward(self):
        return self.function(self.x)


class TestFunction:
    name = ""
    dim = 2
    bounds = (-5.0, 5.0)
    f_star = 0.0
    shift_fraction = 0.0

    def __init__(self, seed=None):
        self.shift = self._make_shift(seed)

    def _make_shift(self, seed):
        if seed is None or self.shift_fraction == 0.0:
            return torch.zeros(self.dim)
        generator = torch.Generator().manual_seed(1000 + seed)
        span = (self.bounds[1] - self.bounds[0]) * self.shift_fraction
        return (torch.rand(self.dim, generator=generator) * 2 - 1) * span

    def __call__(self, x):
        raise NotImplementedError

    def x_star(self):
        raise NotImplementedError


class Sphere(TestFunction):
    name = "sphere"
    bounds = (-5.0, 5.0)
    shift_fraction = 0.3

    def __call__(self, x):
        z = x - self.shift
        return torch.sum(z ** 2)

    def x_star(self):
        return self.shift.clone()


class Rosenbrock(TestFunction):
    name = "rosenbrock"
    bounds = (-2.0, 3.0)
    shift_fraction = 0.0

    def __call__(self, x):
        return torch.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)

    def x_star(self):
        return torch.ones(self.dim)


class Rastrigin(TestFunction):
    name = "rastrigin"
    bounds = (-4.5, 4.5)
    shift_fraction = 0.3

    def __call__(self, x):
        z = x - self.shift
        n = z.shape[0]
        return 10.0 * n + torch.sum(z ** 2 - 10.0 * torch.cos(2.0 * math.pi * z))

    def x_star(self):
        return self.shift.clone()


CALIBRATION_FUNCTION_CLASSES = [Sphere, Rosenbrock, Rastrigin]