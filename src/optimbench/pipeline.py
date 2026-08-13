from optimbench.config import CALIBRATION, TARGETS, TUNING, RUN
from optimbench.registry import resolve_optimizer, param_spec, default_kwargs
from optimbench.calibration import calibrate
from optimbench.tuning import tune_optimizer
from optimbench.targets import calibrate_targets
from optimbench.metrics import steps_to_targets, aggregate
from optimbench.cost import cost_multiplier


def train_eval_wrapper(task, optimizer_name, kwargs, seed, step_budget):
    cls = resolve_optimizer(optimizer_name)
    history = task.run(cls, kwargs, step_budget, seed)
    metric = history[-1][1] if history else (0.0 if task.higher_is_better else float("inf"))
    nfe = history[-1][0] if history else step_budget
    return metric, nfe


def run_pair(task, optimizer_name, adamw_targets=None):
    spec = param_spec(optimizer_name)
    calib = calibrate(optimizer_name, spec, CALIBRATION)

    cm = cost_multiplier(
        lambda: task.build_model(0),
        resolve_optimizer(optimizer_name), default_kwargs(spec),
        resolve_optimizer("adamw"), {},
        steps=20,
    )

    def train_eval_fn(name, kwargs, trial_number):
        seed = TUNING.sampler_seed
        metric, nfe = train_eval_wrapper(task, name, kwargs, seed, task.max_steps_cap // 4)
        return (metric if task.higher_is_better else -metric), nfe

    results = {}
    for budget_name, n_trials in TUNING.budgets.items():
        best_kwargs, tuning_nfe = tune_optimizer(
            optimizer_name, spec, calib["bounds"], calib["fixed"],
            train_eval_fn, n_trials, TUNING.sampler_seed
        )
        cls = resolve_optimizer(optimizer_name)
        histories, train_nfe_total = [], 0
        for seed in range(RUN.final_seeds):
            history = task.run(cls, best_kwargs, task.max_steps_cap, seed)
            histories.append(history)
            train_nfe_total += history[-1][0] if history else task.max_steps_cap

        if adamw_targets is None and optimizer_name == "adamw" and budget_name == "medium":
            adamw_targets = calibrate_targets(histories, TARGETS.levels, TARGETS.max_budget_multiplier, task.higher_is_better)

        targets = adamw_targets["targets"] if adamw_targets else None
        steps_matrix = [steps_to_targets(h, targets, task.higher_is_better) for h in histories] if targets else []
        per_target = list(zip(*steps_matrix)) if steps_matrix else []

        results[budget_name] = {
            "hyperparams": best_kwargs,
            "tuning_nfe": tuning_nfe * cm,
            "train_nfe": train_nfe_total * cm,
            "total_nfe": (tuning_nfe + train_nfe_total) * cm,
            "final_metric": aggregate([h[-1][1] for h in histories if h]),
            "steps_to_targets": [aggregate(list(s)) for s in per_target],
        }

    return results, adamw_targets
