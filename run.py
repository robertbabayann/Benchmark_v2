import json
import dataclasses
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from settings import BENCHMARK, CALIBRATION_RUN, COST, FAST_TEST, GOAL


STAGES = [
    ("Fast Test", "run_fast_test", FAST_TEST),
    ("Cost Evaluation", None, COST),
    ("Search Space Calibration", "run_calibration", CALIBRATION_RUN),
    ("Goal Calibration", "run_goal_calibration", GOAL),
]

TRUE_WORDS = {"y", "yes", "true", "1"}
FALSE_WORDS = {"n", "no", "false", "0"}


def parse_value(current, raw):
    if isinstance(current, bool):
        lowered = raw.lower()
        if lowered in TRUE_WORDS:
            return True
        if lowered in FALSE_WORDS:
            return False
        raise ValueError(raw)
    if current is None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    if isinstance(current, tuple):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if isinstance(current, dict):
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(raw)
        return value
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def edit_settings(settings):
    print("Press Enter to keep the current value")
    for f in dataclasses.fields(settings):
        current = getattr(settings, f.name)
        while True:
            raw = input(f"  {f.name} [{current}]: ").strip()
            if not raw:
                break
            try:
                setattr(settings, f.name, parse_value(current, raw))
                break
            except (ValueError, json.JSONDecodeError):
                print("    invalid value, try again")


def prepare(settings):
    answer = input("\nEdit settings? [Y/N]: ").strip().lower()
    if answer in TRUE_WORDS:
        edit_settings(settings)


def run_stage(label, module_name, settings):
    print(f"\n{label}:")
    prepare(settings)
    try:
        module = importlib.import_module(module_name)
        module.main()
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Exception as e:
        print(f"\nfailed: {e}")


def parse_indices(raw, limit):
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(part)
        value = int(part)
        if not 1 <= value <= limit:
            raise ValueError(part)
        indices.append(value)
    return indices


def task_select_menu(title, tasks, all_label):
    full_index = len(tasks) + 1
    print(f"\n{title}")
    for index, name in enumerate(tasks, 1):
        print(f"  {index}) {name}")
    print(f"  {full_index}) {all_label}")
    print("  0) Back")

    raw = input(f"Select tasks (e.g. 1,3 or {full_index}): ").strip()
    if raw == "0":
        return None
    try:
        indices = parse_indices(raw, full_index)
    except ValueError:
        print("Invalid choice")
        return None
    if full_index in indices:
        return ()
    return tuple(tasks[index - 1] for index in indices)


def benchmark_menu():
    import common

    chosen = task_select_menu("Benchmark:", list(common.TASK_TABLE), "full benchmark")
    if chosen is None:
        return
    BENCHMARK.tasks = chosen
    label = "All tasks" if not chosen else ", ".join(chosen)
    run_stage(f"Benchmark: {label}", "run_benchmark", BENCHMARK)


def cost_menu():
    import common

    chosen = task_select_menu("Cost Evaluation:", list(common.TASK_TABLE), "all tasks")
    if chosen is None:
        return
    COST.tasks = chosen
    label = "All tasks" if not chosen else ", ".join(chosen)
    run_stage(f"Cost Evaluation: {label}", "run_cost", COST)


def show_menu():
    print("\nOptimBench:")
    print("  1) Benchmark")
    for index, (label, _, _) in enumerate(STAGES, 2):
        print(f"  {index}) {label}")
    print(f"  {len(STAGES) + 2}) Analyze Results")
    print("  0) Exit")


def analyze_results():
    try:
        module = importlib.import_module("run_report")
        module.main()
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Exception as e:
        print(f"\nfailed: {e}")


def main():
    while True:
        show_menu()
        raw = input("Select: ").strip()
        if raw == "0":
            break
        if not raw.isdigit():
            continue
        choice = int(raw)
        if choice == 1:
            benchmark_menu()
        elif choice == len(STAGES) + 2:
            analyze_results()
        elif 2 <= choice <= len(STAGES) + 1:
            label, module_name, settings = STAGES[choice - 2]
            if module_name is None:
                cost_menu()
            else:
                run_stage(label, module_name, settings)


if __name__ == "__main__":
    main()
