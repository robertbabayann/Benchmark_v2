import numpy as np


def first_step_to_reach(history, threshold, higher_is_better):
    for step, value in history:
        if (value >= threshold) if higher_is_better else (value <= threshold):
            return step
    return history[-1][0] if history else None


def calibrate_targets(histories, levels, max_budget_multiplier, higher_is_better):
    finals = [h[-1][1] for h in histories if h]
    reference = float(np.median(finals))
    targets = [reference * level for level in levels] if higher_is_better else [reference / level for level in levels]
    target_top = targets[-1]
    steps_to_top = [first_step_to_reach(h, target_top, higher_is_better) for h in histories]
    steps_to_top = [s for s in steps_to_top if s is not None]
    max_budget = max_budget_multiplier * float(np.median(steps_to_top))
    return {"reference": reference, "targets": targets, "max_budget": max_budget}
