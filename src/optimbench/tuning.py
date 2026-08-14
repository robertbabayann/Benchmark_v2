import optuna

from optimbench.registry import expand_flat_params as build_kwargs


def suggest_params(trial, bounds, param_spec):
    point = {}
    for name, (low, high) in bounds.items():
        _, kind, _ = param_spec[name]
        if kind == "log" and low > 0:
            point[name] = trial.suggest_float(name, low, high, log=True)
        else:
            point[name] = trial.suggest_float(name, low, high)
    return point


def tune_optimizer(optimizer_name, param_spec, bounds, fixed, train_eval_fn, n_trials, sampler_seed):
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_nfe = {"total": 0}

    def objective(trial):
        point = suggest_params(trial, bounds, param_spec)
        point.update(fixed)
        kwargs = build_kwargs(point)
        metric, nfe = train_eval_fn(optimizer_name, kwargs, trial.number)
        trial_nfe["total"] += nfe
        return metric

    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials)
        best_point = dict(study.best_trial.params)
        best_point.update(fixed)
    else:
        best_point = {name: param_spec[name][0] for name in param_spec}

    return build_kwargs(best_point), trial_nfe["total"]
