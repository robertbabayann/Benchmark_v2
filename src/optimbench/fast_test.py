import torch

from optimbench.registry import full_registry, param_spec, default_kwargs, build_optimizer


def try_optimizer(name, steps=5, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(10, 1)
    spec = param_spec(name)
    kwargs = default_kwargs(spec)
    opt = build_optimizer(name, list(model.parameters()), kwargs, model=model, num_iterations=steps)
    for _ in range(steps):
        opt.zero_grad()
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    for p in model.parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError("nan or inf after step")


def run_fast_test(names=None, steps=5, seed=0):
    working, failing = [], {}
    for name in (list(names) if names else full_registry()):
        try:
            try_optimizer(name, steps=steps, seed=seed)
            working.append(name)
        except Exception as e:
            failing[name] = str(e)
    return working, failing
