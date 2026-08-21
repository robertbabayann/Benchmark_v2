from optimbench.device import resolve_device
from optimbench.pipeline import run_pair, run_reference

from common import ensure_data, resolve_task, save_json
from settings import PIPELINE


def main():
    task = resolve_task(PIPELINE.task)()
    task.device = resolve_device(PIPELINE.device)
    ensure_data([PIPELINE.task])

    print(f"stage 1: reference '{PIPELINE.reference_name}' on '{PIPELINE.task}' (max_steps_cap={task.max_steps_cap}, device={task.device})")
    reference = run_reference(
        task,
        reference_name=PIPELINE.reference_name,
        n_trials=PIPELINE.reference_trials,
        final_seeds=PIPELINE.final_seeds,
        sampler_seed=PIPELINE.sampler_seed,
    )
    print(f"targets: {reference['targets']}")
    print(f"max_budget: {reference['max_budget']}")

    print(f"\nstage 2: '{PIPELINE.optimizer}' on '{PIPELINE.task}' (step_budget={round(reference['max_budget'])})")
    results = run_pair(
        task,
        PIPELINE.optimizer,
        reference,
        budgets=PIPELINE.budgets,
        final_seeds=PIPELINE.final_seeds,
        sampler_seed=PIPELINE.sampler_seed,
    )
    for budget, data in results.items():
        print(f"\n{budget}:")
        for key, value in data.items():
            print(f"  {key}: {value}")

    if PIPELINE.output_path:
        save_json(PIPELINE.output_path, {"task": PIPELINE.task, "reference": reference, "results": results})
        print(f"\nresults saved to {PIPELINE.output_path}")
    return reference, results


if __name__ == "__main__":
    main()
