import numpy as np


def steps_to_targets(history, targets, higher_is_better):
    result = []
    for target in targets:
        found = None
        for step, value in history:
            if (value >= target) if higher_is_better else (value <= target):
                found = step
                break
        result.append(found)
    return result


def aggregate(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(clean)), "std": float(np.std(clean))}


def performance_ratios(table):
    ratios = {}
    for task_key, scores in table.items():
        valid = {s: v for s, v in scores.items() if v is not None}
        if not valid:
            continue
        best = min(valid.values())
        for solver, value in scores.items():
            ratios.setdefault(solver, {})[task_key] = (value / best) if value is not None else None
    return ratios


def performance_profile(ratios, taus, r_m):
    profiles = {}
    for solver, per_task in ratios.items():
        values = [v if v is not None else r_m for v in per_task.values()]
        n = len(values)
        profiles[solver] = [sum(1 for v in values if v <= tau) / n for tau in taus]
    return profiles
