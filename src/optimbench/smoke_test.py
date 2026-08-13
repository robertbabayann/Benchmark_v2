import torch

from optimbench.registry import full_registry, resolve_optimizer, param_spec, default_kwargs


def try_optimizer(name):
    model = torch.nn.Linear(10, 1)
    cls = resolve_optimizer(name)
    spec = param_spec(name)
    kwargs = default_kwargs(spec)
    try:
        opt = cls(model.parameters(), **kwargs)
    except TypeError:
        opt = cls(model.parameters())
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
