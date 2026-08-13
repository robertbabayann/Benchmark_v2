import math
import torch


class TestFunction:
    name = ""
    dim = 2
    bounds = (-5.0, 5.0)
    f_star = 0.0

    def __call__(self, x):
        raise NotImplementedError

    def x_star(self):
        raise NotImplementedError


class Sphere(TestFunction):
    name = "sphere"
    bounds = (-5.0, 5.0)

    def __call__(self, x):
        return torch.sum(x ** 2)

    def x_star(self):
        return torch.zeros(self.dim)


class Rosenbrock(TestFunction):
    name = "rosenbrock"
    bounds = (-2.0, 3.0)

    def __call__(self, x):
        return torch.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)

    def x_star(self):
        return torch.ones(self.dim)


class Rastrigin(TestFunction):
    name = "rastrigin"
    bounds = (-4.5, 4.5)

    def __call__(self, x):
        n = x.shape[0]
        return 10.0 * n + torch.sum(x ** 2 - 10.0 * torch.cos(2.0 * math.pi * x))

    def x_star(self):
        return torch.zeros(self.dim)


CALIBRATION_FUNCTIONS = [Sphere(), Rosenbrock(), Rastrigin()]
