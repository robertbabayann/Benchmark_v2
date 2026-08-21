from optimbench.device import resolve_device
from optimbench.pipeline import run_pair, run_reference

from common import resolve_task, save_json, select_optimizers, select_tasks
from settings import BENCHMARK


def run_task(task_name, optimizer_names, device):
    task = resolve_task(task_name)()
    task.device = device

    print(f"[{task_name}] stage 1: reference '{BENCHMARK.reference_name}' (max_steps_cap={task.max_steps_cap}, device={device})")
    reference = run_reference(
        task,
        reference_name=BENCHMARK.reference_name,
        n_trials=BENCHMARK.reference_trials,
        final_seeds=BENCHMARK.final_seeds,
        sampler_seed=BENCHMARK.sampler_seed,
    )
    print(f"[{task_name}] targets: {reference['targets']}, max_budget: {reference['max_budget']}")

    payload = {"reference": reference, "step_budget": round(reference["max_budget"]), "optimizers": {}}
    step_budget = round(reference["max_budget"])
    for name in optimizer_names:
        print(f"[{task_name}] stage 2: '{name}' (step_budget={step_budget})")
        payload["optimizers"][name] = run_pair(
            task,
            name,
            reference,
            budgets=BENCHMARK.budgets,
            final_seeds=BENCHMARK.final_seeds,
            sampler_seed=BENCHMARK.sampler_seed,
        )
    return payload


def main():
    tasks = select_tasks(BENCHMARK.tasks)
    optimizers = select_optimizers(BENCHMARK.optimizers)
    device = resolve_device(BENCHMARK.device)
    print(
        f"benchmark: {len(tasks)} tasks x {len(optimizers)} optimizers on {device} "
        f"(budgets={list(BENCHMARK.budgets)}, seeds={BENCHMARK.final_seeds}) -> {BENCHMARK.output_path}"
    )
    summary = {}
    for index, task_name in enumerate(tasks, 1):
        print(f"\n=== task {index}/{len(tasks)}: {task_name} ===")
        summary[task_name] = run_task(task_name, optimizers, device)
        save_json(BENCHMARK.output_path, summary)

    print(f"\ndone, results saved to {BENCHMARK.output_path}")
    return summary


if __name__ == "__main__":
    main()
