import json
from pathlib import Path

import numpy as np

from optimbench.cost import measure_step_time
from optimbench.device import resolve_device
from optimbench.registry import default_kwargs, param_spec

from common import ROOT, resolve_task, save_json, select_optimizers, select_tasks
from settings import COST


def load_previous(output_path):
    if output_path.exists():
        try:
            data = json.loads(output_path.read_text())
            return data.get("results", {}), data.get("failed", {})
        except json.JSONDecodeError:
            print("warning: corrupted checkpoint ignored")
    return {}, {}


def prune_resolved(failed, results):
    pruned = {}
    for key, reason in failed.items():
        name, _, task = key.partition(":")
        entry = results.get(name)
        if entry and (not task or task in entry.get("tasks", {})):
            continue
        pruned[key] = reason
    return pruned


def build_payload(results, failed):
    baseline_tasks = results.get(COST.baseline, {}).get("tasks", {})
    for entry in results.values():
        times = [t["step_time_ms"] for t in entry["tasks"].values()]
        entry["step_time_ms"] = float(np.mean(times)) if times else None
        shared = [tn for tn in entry["tasks"] if tn in baseline_tasks]
        entry["cost_multiplier"] = (
            float(np.mean([entry["tasks"][tn]["step_time_ms"] / baseline_tasks[tn]["step_time_ms"] for tn in shared]))
            if shared else None
        )
    ordered = sorted(
        results.items(),
        key=lambda item: item[1]["cost_multiplier"] if item[1]["cost_multiplier"] is not None else float("inf"),
    )
    return {
        "timing_steps": COST.timing_steps,
        "seeds": COST.seeds,
        "baseline": COST.baseline,
        "results": dict(ordered),
        "failed": prune_resolved(failed, results),
    }


def print_table(payload, tasks):
    results = payload["results"]
    function_header = "".join(f"{task:>16}" for task in tasks)
    print(f"\n{'optimizer':<24}{'x_' + COST.baseline:>10}{'' :>10}{function_header}")
    for name, entry in results.items():
        multiplier = f"{entry['cost_multiplier']:.2f}" if entry['cost_multiplier'] is not None else "-"
        step_ms = f"{entry['step_time_ms']:>10.3f}" if entry['step_time_ms'] is not None else f"{'-':>10}"
        row = f"{name:<24}{multiplier:>10}{step_ms}"
        row += "".join(
            f"{entry['tasks'][task]['step_time_ms']:>16.3f}" if task in entry["tasks"] else f"{'-':>16}"
            for task in tasks
        )
        print(row)


def main():
    tasks = select_tasks(COST.tasks)
    names = select_optimizers(COST.optimizers)
    if COST.baseline not in names:
        raise ValueError(f"baseline '{COST.baseline}' is not in the selected optimizers")

    output_path = Path(COST.output_path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    results, failed = load_previous(output_path)
    device = resolve_device(COST.device)
    results = {
        name: entry
        for name, entry in results.items()
        if isinstance(entry, dict) and "kwargs" in entry and isinstance(entry.get("tasks"), dict)
    }

    print(
        f"cost: {len(names)} optimizers x {len(tasks)} tasks on {device}, "
        f"{COST.seeds} seeds x {COST.timing_steps} steps, baseline '{COST.baseline}'"
    )
    if results or failed:
        print(f"resuming: found {len(results)} measured optimizers, {len(failed)} recorded failures")

    interrupted = False
    try:
        for name in names:
            entry = results.get(name)
            missing = [
                task_name
                for task_name in tasks
                if (not entry or task_name not in entry["tasks"])
                and (COST.retry_failed or f"{name}:{task_name}" not in failed)
            ]
            if not missing:
                print(f"  skipped (already measured): {name}")
                continue

            if name not in results:
                try:
                    kwargs = default_kwargs(param_spec(name))
                except Exception as e:
                    failed[name] = str(e)
                    print(f"  spec failed: {name}: {e}")
                    continue
                results[name] = {"kwargs": kwargs, "tasks": {}}

            kwargs = results[name]["kwargs"]
            for task_name in missing:
                task_factory = resolve_task(task_name)

                def model_fn(seed, factory=task_factory):
                    return factory().build_model(seed).to(device)

                try:
                    times = [
                        measure_step_time(model_fn(seed), name, kwargs, COST.timing_steps)
                        for seed in range(COST.seeds)
                    ]
                    results[name]["tasks"][task_name] = {
                        "step_time_ms": float(np.mean(times)) * 1000.0,
                        "step_time_std_ms": float(np.std(times)) * 1000.0,
                    }
                    print(f"  measured: {name} on {task_name}")
                except Exception as e:
                    failed[f"{name}:{task_name}"] = str(e)
                    print(f"  failed:   {name} on {task_name}: {e}")

                save_json(COST.output_path, build_payload(results, failed))
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted, progress saved")

    payload = build_payload(results, failed)
    print_table(payload, tasks)
    if payload["failed"]:
        print(f"\nfailed pairs: {len(payload['failed'])}")
    save_json(COST.output_path, payload)

    status = "partially complete (interrupted)" if interrupted else "done"
    print(f"\n{status}, results saved to {COST.output_path}")
    return payload


if __name__ == "__main__":
    main()
