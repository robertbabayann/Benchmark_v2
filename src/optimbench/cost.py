import time
import torch

from optimbench.registry import build_optimizer


def measure_step_time(model, optimizer_name, kwargs, steps):
    optimizer = build_optimizer(optimizer_name, list(model.parameters()), kwargs, model=model, num_iterations=steps)
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


def cost_multiplier(model_fn, optimizer_name, kwargs, baseline_name, baseline_kwargs, steps):
    target_time = measure_step_time(model_fn(), optimizer_name, kwargs, steps)
    baseline_time = measure_step_time(model_fn(), baseline_name, baseline_kwargs, steps)
    return target_time / baseline_time