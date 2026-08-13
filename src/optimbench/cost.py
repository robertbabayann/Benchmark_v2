import time
import torch


def measure_step_time(model, optimizer_cls, kwargs, steps):
    optimizer = optimizer_cls(model.parameters(), **kwargs)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def cost_multiplier(model_fn, optimizer_cls, kwargs, baseline_cls, baseline_kwargs, steps):
    target_time = measure_step_time(model_fn(), optimizer_cls, kwargs, steps)
    baseline_time = measure_step_time(model_fn(), baseline_cls, baseline_kwargs, steps)
    return target_time / baseline_time
