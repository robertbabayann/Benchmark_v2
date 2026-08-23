from optimbench.config import CALIBRATION, TARGETS, TUNING, RUN
from optimbench.registry import param_spec, default_kwargs
from optimbench.presets import get_calibration, get_cost_multiplier
from optimbench.tuning import tune_optimizer
from optimbench.targets import calibrate_targets
from optimbench.metrics import steps_to_targets, aggregate
from optimbench.cost import cost_multiplier


def train_eval_wrapper(task, optimizer_name, kwargs, seed, step_budget):
    history = task.run(optimizer_name, kwargs, step_budget, seed)
    metric = history[-1][1] if history else (0.0 if task.higher_is_better else float("inf"))
    nfe = history[-1][0] if history else step_budget
    return metric, nfe


def run_reference(task, reference_name="adamw", n_trials=None, final_seeds=None, sampler_seed=None):
    spec = param_spec(reference_name)
    calib = get_calibration(reference_name, spec, CALIBRATION)
    tuning_seed = TUNING.sampler_seed if sampler_seed is None else sampler_seed
    trials = TUNING.budgets["medium"] if n_trials is None else n_trials

    def train_eval_fn(name, kwargs, trial_number):
        metric, nfe = train_eval_wrapper(task, name, kwargs, tuning_seed, task.max_steps_cap // 4)
        return (metric if task.higher_is_better else -metric), nfe

    best_kwargs, _ = tune_optimizer(
        reference_name, spec, calib["active_bounds"], calib["fixed"],
        train_eval_fn, trials, tuning_seed
    )

    seeds = RUN.final_seeds if final_seeds is None else final_seeds
    histories = [
        task.run(reference_name, best_kwargs, task.max_steps_cap, seed)
        for seed in range(seeds)
    ]
    return calibrate_targets(histories, TARGETS.levels, TARGETS.max_budget_multiplier, task.higher_is_better)


def run_pair(task, optimizer_name, reference, budgets=None, final_seeds=None, sampler_seed=None, task_name=None):
    spec = param_spec(optimizer_name)
    calib = get_calibration(optimizer_name, spec, CALIBRATION)
    budget_map = TUNING.budgets if budgets is None else budgets
    seeds_count = RUN.final_seeds if final_seeds is None else final_seeds
    tuning_seed = TUNING.sampler_seed if sampler_seed is None else sampler_seed

    task_key = task_name or getattr(task, "name", "")
    cm = get_cost_multiplier(optimizer_name, task_key)
    if cm is None:
        print(
            f"warning: no cost preset for '{optimizer_name}' on '{task_key}', "
            f"measuring step time now (20 steps)"
        )
        cm = cost_multiplier(
            lambda: task.build_model(0),
            optimizer_name, default_kwargs(spec),
            "adamw", {},
            steps=20,
        )

    def train_eval_fn(name, kwargs, trial_number):
        metric, nfe = train_eval_wrapper(task, name, kwargs, tuning_seed, task.max_steps_cap // 4)
        return (metric if task.higher_is_better else -metric), nfe

    results = {}
    step_budget = int(round(reference["max_budget"]))
    targets = reference["targets"]

    for budget_name, n_trials in budget_map.items():
        best_kwargs, tuning_nfe = tune_optimizer(
            optimizer_name, spec, calib["active_bounds"], calib["fixed"],
            train_eval_fn, n_trials, tuning_seed
        )
        histories, train_nfe_total = [], 0
        for seed in range(seeds_count):
            history = task.run(optimizer_name, best_kwargs, step_budget, seed)
            histories.append(history)
            train_nfe_total += history[-1][0] if history else step_budget

        steps_matrix = [steps_to_targets(h, targets, task.higher_is_better) for h in histories]
        per_target = list(zip(*steps_matrix))

        results[budget_name] = {
            "hyperparams": best_kwargs,
            "tuning_nfe": tuning_nfe * cm,
            "train_nfe": train_nfe_total * cm,
            "total_nfe": (tuning_nfe + train_nfe_total) * cm,
            "final_metric": aggregate([h[-1][1] for h in histories if h]),
            "steps_to_targets": [aggregate(list(s)) for s in per_target],
        }

    return results
