import math
import numpy as np
import torch
from scipy.stats.qmc import Sobol
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from optimbench.registry import build_optimizer, expand_flat_params as build_kwargs
from optimbench.testfunctions import CALIBRATION_FUNCTION_CLASSES


CALIBRATION_INCOMPATIBLE = {"adammini"}

UNIVERSAL_POLICY = {
    "lr": (1e-5, 1e-1),
    "weight_decay": (1e-6, 1e-2),
    "betas_0": (0.8, 0.99),
    "betas_1": (0.9, 0.999),
}


def sample_hyperparams(param_spec, n_points, log_span, decay_span, decay_cap, unit_eps, seed):
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
            elif kind == "decay":
                exponent = (row[i] * 2.0 - 1.0) * decay_span
                point[name] = min(anchor * (10.0 ** exponent), decay_cap)
            else:
                point[name] = unit_eps + row[i] * (1.0 - 2.0 * unit_eps)
        points.append(point)
    return points


def run_single(optimizer_name, flat_point, test_function, steps, seed, divergence_threshold):
    torch.manual_seed(seed)
    x = torch.empty(test_function.dim).uniform_(*test_function.bounds).requires_grad_(True)
    kwargs = build_kwargs(flat_point)
    try:
        opt = build_optimizer(optimizer_name, [x], kwargs, model=None, num_iterations=steps)
    except Exception:
        return "init_error", None

    try:
        for _ in range(steps):
            opt.zero_grad()
            loss = test_function(x)
            if not torch.isfinite(loss):
                return "diverged", None
            loss.backward()
            opt.step()
            if not torch.isfinite(x).all():
                return "diverged", None
    except Exception:
        return "diverged", None

    with torch.no_grad():
        final = test_function(x).item()
    if not np.isfinite(final) or abs(final) > divergence_threshold:
        return "diverged", None
    return "converged", abs(final - test_function.f_star)


def calibrate_optimizer(optimizer_name, points, function_classes, seeds_per_point, steps, divergence_threshold):
    records = []
    for point_id, point in enumerate(points):
        for func_cls in function_classes:
            for seed in range(seeds_per_point):
                func = func_cls(seed=seed)
                status, gap = run_single(optimizer_name, point, func, steps, seed, divergence_threshold)
                records.append({
                    "point_id": point_id, "point": point, "function": func.name,
                    "seed": seed, "status": status, "gap": gap,
                })
    return records


def compute_bounds(records, param_spec, low_percentile, high_percentile, decay_cap, log_clip, decay_clip):
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
            if kind == "decay":
                bounds[name] = (anchor / decay_clip, min(anchor * decay_clip, decay_cap))
            elif kind == "log":
                bounds[name] = (anchor / 10.0, anchor * 10.0)
            else:
                bounds[name] = (0.0, 1.0)
            continue
        lo = float(np.percentile(values, low_percentile))
        hi = float(np.percentile(values, high_percentile))
        if kind == "log":
            lo = max(lo, anchor / log_clip)
            hi = min(hi, anchor * log_clip)
        elif kind == "decay":
            lo = max(lo, anchor / decay_clip)
            hi = min(hi, min(anchor * decay_clip, decay_cap))
        bounds[name] = (lo, hi) if lo < hi else (lo, lo + 1e-8)
    return bounds


def record_output(status, gap, penalty):
    return math.log10(gap + 1e-12) if status == "converged" else penalty


def build_training_table(records, names, bounds, expansion, penalty):
    expanded_bounds = {}
    for name in names:
        lo, hi = bounds[name]
        margin = (hi - lo) * (expansion - 1.0) / 2.0
        expanded_bounds[name] = (lo - margin, hi + margin)

    grouped = {}
    for record in records:
        grouped.setdefault(record["point_id"], []).append(record)

    rows, targets = [], []
    for point_id, group in grouped.items():
        point = group[0]["point"]
        inside = all(expanded_bounds[name][0] <= point[name] <= expanded_bounds[name][1] for name in names)
        if not inside:
            continue
        outputs = [record_output(r["status"], r["gap"], penalty) for r in group]
        rows.append([point[name] for name in names])
        targets.append(float(np.mean(outputs)))

    if not rows:
        return None, None
    return np.array(rows), np.array(targets)


def fit_surrogate(X, y, seed, cv_folds):
    if len(X) < cv_folds * 3:
        return None, 0.0
    model = RandomForestRegressor(n_estimators=200, random_state=seed)
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
    model.fit(X, y)
    return model, float(np.mean(scores))


def build_problem(names, bounds):
    return {"num_vars": len(names), "names": names, "bounds": [list(bounds[n]) for n in names]}


def sobol_base_n(n_samples):
    m = math.log2(n_samples)
    if not m.is_integer():
        raise ValueError("sensitivity_samples must be a power of two")
    return n_samples


def run_sensitivity_surrogate(param_spec, bounds, model, n_samples, seed):
    names = list(param_spec.keys())
    problem = build_problem(names, bounds)
    samples = sobol_sample.sample(problem, sobol_base_n(n_samples), calc_second_order=True, seed=seed)
    outputs = model.predict(samples)
    indices = sobol_analyze.analyze(problem, outputs, calc_second_order=True, seed=seed)
    return {"names": names, "S1": indices["S1"].tolist(), "ST": indices["ST"].tolist()}


def run_sensitivity_real(optimizer_name, param_spec, bounds, function_classes, steps, seeds_per_point, n_samples, seed, penalty):
    names = list(param_spec.keys())
    problem = build_problem(names, bounds)
    samples = sobol_sample.sample(problem, sobol_base_n(n_samples), calc_second_order=True, seed=seed)
    outputs = np.zeros(samples.shape[0])
    for i, row in enumerate(samples):
        flat_point = {name: float(value) for name, value in zip(names, row)}
        results = []
        for func_cls in function_classes:
            for s in range(seeds_per_point):
                func = func_cls(seed=s)
                status, gap = run_single(optimizer_name, flat_point, func, steps, s, divergence_threshold=1e6)
                results.append(record_output(status, gap, penalty))
        outputs[i] = float(np.mean(results))
    indices = sobol_analyze.analyze(problem, outputs, calc_second_order=True, seed=seed)
    return {"names": names, "S1": indices["S1"].tolist(), "ST": indices["ST"].tolist()}


def calibrate_from_universal_policy(optimizer_name, param_spec):
    names = list(param_spec.keys())
    all_bounds = {}
    active = []
    for name in names:
        default, kind, anchor = param_spec[name]
        if name in UNIVERSAL_POLICY:
            all_bounds[name] = UNIVERSAL_POLICY[name]
            active.append(name)
        else:
            all_bounds[name] = (anchor, anchor)
    tuned_bounds = {n: all_bounds[n] for n in active}
    fixed = {n: param_spec[n][0] for n in names if n not in active}
    sensitivity = {"names": names, "S1": [None] * len(names), "ST": [None] * len(names)}
    return {
        "active_bounds": tuned_bounds,
        "all_bounds": all_bounds,
        "fixed": fixed,
        "sensitivity": sensitivity,
        "records": [],
        "surrogate_used": False,
        "surrogate_r2": None,
        "calibration_skipped": True,
        "skip_reason": "structurally incompatible with synthetic test functions, using universal policy",
    }


def calibrate(optimizer_name, param_spec, cfg, seed=0):
    if optimizer_name in CALIBRATION_INCOMPATIBLE:
        return calibrate_from_universal_policy(optimizer_name, param_spec)

    penalty = 6.0
    points = sample_hyperparams(
        param_spec, cfg.sobol_points, cfg.log_scale_span, cfg.decay_log_scale_span,
        cfg.decay_hard_cap, cfg.unit_eps, seed
    )
    records = calibrate_optimizer(
        optimizer_name, points, CALIBRATION_FUNCTION_CLASSES, cfg.seeds_per_point, cfg.steps_per_run, cfg.divergence_threshold
    )
    all_bounds = compute_bounds(
        records, param_spec, cfg.bound_low_percentile, cfg.bound_high_percentile,
        cfg.decay_hard_cap, cfg.log_clip_multiplier, cfg.decay_clip_multiplier
    )

    names = list(param_spec.keys())
    surrogate_used, r2 = False, None

    if cfg.use_surrogate:
        X, y = build_training_table(records, names, all_bounds, cfg.surrogate_bounds_expansion, penalty)
        if X is not None and len(X) >= cfg.surrogate_min_samples:
            model, r2 = fit_surrogate(X, y, seed, cfg.surrogate_cv_folds)
            if model is not None and r2 >= cfg.surrogate_min_r2:
                sensitivity = run_sensitivity_surrogate(param_spec, all_bounds, model, cfg.sensitivity_samples, seed)
                surrogate_used = True

    if not surrogate_used:
        sensitivity = run_sensitivity_real(
            optimizer_name, param_spec, all_bounds, CALIBRATION_FUNCTION_CLASSES,
            cfg.steps_per_run, cfg.seeds_per_point, cfg.sensitivity_samples, seed, penalty
        )

    active = [
        n for n, st in zip(sensitivity["names"], sensitivity["ST"])
        if st is not None and not math.isnan(st) and st >= cfg.sensitivity_threshold
    ]
    tuned_bounds = {n: all_bounds[n] for n in active}
    fixed = {n: param_spec[n][0] for n in param_spec if n not in active}

    return {
        "active_bounds": tuned_bounds,
        "all_bounds": all_bounds,
        "fixed": fixed,
        "sensitivity": sensitivity,
        "records": records,
        "surrogate_used": surrogate_used,
        "surrogate_r2": r2,
        "calibration_skipped": False,
        "skip_reason": None,
    }