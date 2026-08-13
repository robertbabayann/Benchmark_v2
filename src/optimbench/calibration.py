import numpy as np
import torch
from scipy.stats.qmc import Sobol
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from optimbench.registry import resolve_optimizer
from optimbench.testfunctions import CALIBRATION_FUNCTIONS


def sample_hyperparams(param_spec, n_points, log_span, unit_eps, seed):
    names = list(param_spec.keys())
    d = len(names)
    sampler = Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n_points, 1))))
    u = sampler.random_base2(m)[:n_points]
    points = []
    for row in u:
        point = {}
        for i, name in enumerate(names):
            default, kind, anchor = param_spec[name]
            if kind == "log":
                exponent = (row[i] * 2.0 - 1.0) * log_span
                point[name] = anchor * (10.0 ** exponent)
            else:
                point[name] = unit_eps + row[i] * (1.0 - 2.0 * unit_eps)
        points.append(point)
    return points


def build_kwargs(flat_point):
    kwargs = {}
    tuples = {}
    for name, value in flat_point.items():
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            tuples.setdefault(parts[0], {})[int(parts[1])] = value
        else:
            kwargs[name] = value
    for tname, parts in tuples.items():
        kwargs[tname] = tuple(parts[i] for i in sorted(parts))
    return kwargs


def run_single(optimizer_name, flat_point, test_function, steps, seed, divergence_threshold):
    torch.manual_seed(seed)
    x = torch.empty(test_function.dim).uniform_(*test_function.bounds).requires_grad_(True)
    kwargs = build_kwargs(flat_point)
    cls = resolve_optimizer(optimizer_name)
    try:
        opt = cls([x], **kwargs)
    except TypeError:
        return "init_error", None

    for _ in range(steps):
        opt.zero_grad()
        loss = test_function(x)
        if not torch.isfinite(loss):
            return "diverged", None
        loss.backward()
        opt.step()
        if not torch.isfinite(x).all():
            return "diverged", None

    with torch.no_grad():
        final = test_function(x).item()
    if not np.isfinite(final) or abs(final) > divergence_threshold:
        return "diverged", None
    return "converged", abs(final - test_function.f_star)


def calibrate_optimizer(optimizer_name, points, functions, seeds_per_point, steps, divergence_threshold):
    records = []
    for point in points:
        for func in functions:
            for seed in range(seeds_per_point):
                status, gap = run_single(optimizer_name, point, func, steps, seed, divergence_threshold)
                records.append({"point": point, "function": func.name, "seed": seed, "status": status, "gap": gap})
    return records


def compute_bounds(records, param_spec):
    accepted = {name: [] for name in param_spec}
    for record in records:
        if record["status"] != "converged":
            continue
        for name, value in record["point"].items():
            accepted[name].append(value)
    bounds = {}
    for name, values in accepted.items():
        default, kind, anchor = param_spec[name]
        if not values:
            bounds[name] = (anchor * 0.5, anchor * 2.0) if kind == "log" else (0.0, 1.0)
        else:
            bounds[name] = (min(values), max(values))
    return bounds


def run_sensitivity(optimizer_name, param_spec, bounds, functions, steps, seeds_per_point, n_samples, seed):
    names = list(param_spec.keys())
    problem = {"num_vars": len(names), "names": names, "bounds": [list(bounds[n]) for n in names]}
    samples = sobol_sample.sample(problem, n_samples, calc_second_order=True, seed=seed)
    outputs = np.zeros(samples.shape[0])
    for i, row in enumerate(samples):
        flat_point = {name: float(value) for name, value in zip(names, row)}
        gaps = []
        for func in functions:
            for s in range(seeds_per_point):
                status, gap = run_single(optimizer_name, flat_point, func, steps, s, divergence_threshold=1e6)
                gaps.append(np.log10(gap + 1e-12) if status == "converged" else 6.0)
        outputs[i] = float(np.mean(gaps))
    indices = sobol_analyze.analyze(problem, outputs, calc_second_order=True, seed=seed)
    return {"names": names, "S1": indices["S1"].tolist(), "ST": indices["ST"].tolist()}


def calibrate(optimizer_name, param_spec, cfg, seed=0):
    points = sample_hyperparams(param_spec, cfg.sobol_points, cfg.log_scale_span, cfg.unit_eps, seed)
    records = calibrate_optimizer(
        optimizer_name, points, CALIBRATION_FUNCTIONS, cfg.seeds_per_point, cfg.steps_per_run, cfg.divergence_threshold
    )
    bounds = compute_bounds(records, param_spec)
    sensitivity = run_sensitivity(
        optimizer_name, param_spec, bounds, CALIBRATION_FUNCTIONS,
        cfg.steps_per_run, cfg.seeds_per_point, cfg.sensitivity_samples, seed
    )
    active = [n for n, s1 in zip(sensitivity["names"], sensitivity["S1"]) if s1 >= cfg.sensitivity_threshold]
    tuned_bounds = {n: bounds[n] for n in active}
    fixed = {n: param_spec[n][0] for n in param_spec if n not in active}
    return {"bounds": tuned_bounds, "fixed": fixed, "sensitivity": sensitivity, "records": records}
