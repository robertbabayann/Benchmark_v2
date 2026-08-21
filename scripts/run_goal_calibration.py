from optimbench.device import resolve_device
from optimbench.pipeline import run_reference

from common import ensure_data, resolve_task, save_json
from settings import GOAL


def main():
    task = resolve_task(GOAL.task)()
    task.device = resolve_device(GOAL.device)
    ensure_data([GOAL.task])
    print(f"goal calibration: '{GOAL.reference_name}' on '{GOAL.task}' (max_steps_cap={task.max_steps_cap}, device={task.device})")
    reference = run_reference(
        task,
        reference_name=GOAL.reference_name,
        n_trials=GOAL.reference_trials,
        final_seeds=GOAL.final_seeds,
        sampler_seed=GOAL.sampler_seed,
    )
    print(f"targets: {reference['targets']}")
    print(f"max_budget: {reference['max_budget']}")

    if GOAL.output_path:
        save_json(GOAL.output_path, {"task": GOAL.task, "reference": GOAL.reference_name, **reference})
        print(f"\nresults saved to {GOAL.output_path}")
    return reference


if __name__ == "__main__":
    main()
