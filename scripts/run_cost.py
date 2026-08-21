import numpy as np

from optimbench.cost import measure_step_time
from optimbench.registry import default_kwargs, param_spec

from common import resolve_task, save_json, select_optimizers, select_tasks
from settings import COST


def timed_step(name, kwargs, model_fn):
    times = [
        measure_step_time(model_fn(seed), name, kwargs, COST.timing_steps)
        for seed in range(COST.seeds)
    ]
    return float(np.mean(times)), float(np.std(times))


def main():
    tasks = select_tasks(COST.tasks)
    names = select_optimizers(COST.optimizers)
    if COST.baseline not in names:
        raise ValueError(f"baseline '{COST.baseline}' is not in the selected optimizers")

    print(
        f"cost: {len(names)} optimizers x {len(tasks)} tasks (cpu), "
        f"{COST.seeds} seeds x {COST.timing_steps} steps, baseline '{COST.baseline}'"
    )
    results = {}
    failed = {}
    for name in names:
        try:
            kwargs = default_kwargs(param_spec(name))
        except Exception as e:
            failed[name] = str(e)
            print(f"  spec failed: {name}: {e}")
            continue
        per_task = {}
        for task_name in tasks:
            task_factory = resolve_task(task_name)

            def model_fn(seed):
                return task_factory().build_model(seed)

            try:
                mean_time, std_time = timed_step(name, kwargs, model_fn)
                per_task[task_name] = {"step_time_ms": mean_time * 1000.0, "step_time_std_ms": std_time * 1000.0}
                print(f"  measured: {name} on {task_name}")
            except Exception as e:
                failed[f"{name}:{task_name}"] = str(e)
                print(f"  failed:   {name} on {task_name}: {e}")
        if per_task:
            results[name] = {"kwargs": kwargs, "tasks": per_task}

    baseline_tasks = results.get(COST.baseline, {}).get("tasks", {})
    for name, entry in results.items():
        entry["step_time_ms"] = float(np.mean([t["step_time_ms"] for t in entry["tasks"].values()]))
        shared = [tn for tn in entry["tasks"] if tn in baseline_tasks]
        entry["cost_multiplier"] = (
            float(np.mean([entry["tasks"][tn]["step_time_ms"] / baseline_tasks[tn]["step_time_ms"] for tn in shared]))
            if shared else None
        )

    ordered = sorted(results.items(), key=lambda item: item[1]["cost_multiplier"] if item[1]["cost_multiplier"] is not None else float("inf"))
    header = f"{'optimizer':<24}{'x_' + COST.baseline:>10}{'' :>10}" + "".join(f"{task:>16}" for task in tasks)
    print("\n" + header)
    for name, entry in ordered:
        multiplier = f"{entry['cost_multiplier']:.2f}" if entry["cost_multiplier"] is not None else "-"
        row = f"{name:<24}{multiplier:>10}{entry['step_time_ms']:>10.3f}"
        row += "".join(
            f"{entry['tasks'][task]['step_time_ms']:>16.3f}" if task in entry["tasks"] else f"{'-':>16}"
            for task in tasks
        )
        print(row)

    payload = {
        "timing_steps": COST.timing_steps,
        "seeds": COST.seeds,
        "baseline": COST.baseline,
        "results": dict(ordered),
        "failed": failed,
    }
    if COST.output_path:
        save_json(COST.output_path, payload)
        print(f"\nresults saved to {COST.output_path}")
    return payload


if __name__ == "__main__":
    main()
