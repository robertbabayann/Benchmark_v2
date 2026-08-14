import torch

from optimbench.registry import full_registry, param_spec, default_kwargs, build_optimizer, optimizer_source


def try_optimizer(name):
    model = torch.nn.Linear(10, 1)
    spec = param_spec(name)
    kwargs = default_kwargs(spec)
    opt = build_optimizer(name, list(model.parameters()), kwargs, model=model, num_iterations=5)
    for _ in range(5):
        opt.zero_grad()
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    for p in model.parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError("nan or inf after step")


def run_smoke_test():
    working, failing = [], {}
    for name in full_registry():
        try:
            try_optimizer(name)
            working.append(name)
        except Exception as e:
            failing[name] = str(e)
    return working, failing