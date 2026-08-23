"""Compact viewer for presets/cost.json and presets/search_space.json (temporary analysis helper)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRESETS = ROOT / "src" / "optimbench" / "presets"

ALL_OPTIMIZERS = [
    "sgd", "nag", "rmsprop", "adadelta", "adam", "adamw", "amsgrad", "nadam", "radam",
    "adabelief", "diffgrad", "yogi", "lamb", "adan", "lion", "amos", "prodigy",
    "schedulefreeadamw", "ademamix", "adopt", "mars", "shampoo", "soap", "muon",
    "lars", "madgrad", "sm3", "adammini", "lookaheadradam", "ranger21",
]


def load(name):
    path = PRESETS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"warning: corrupted {path.name}")
        return None


def fmt(value, width=None):
    if value is None:
        text = "-"
    elif isinstance(value, str):
        text = value
    elif value == 0:
        text = "0"
    elif abs(value) >= 1000 or abs(value) < 0.001:
        text = f"{value:.1e}"
    else:
        text = f"{value:.4g}"
    return text if width is None else f"{text:>{width}}"


def pct(value):
    return "-" if value is None else f"{value * 100:.0f}%"


def cost_overview(cost):
    results = cost.get("results", {})
    baseline = cost.get("baseline", "?")
    seeds, steps = cost.get("seeds"), cost.get("timing_steps")
    print(f"\n=== COST (baseline={baseline}, {seeds} seeds x {steps} steps) ===")
    header = f"{'optimizer':<18}{'mult':>7}{'mean_ms':>10}{'min_ms':>10}{'max_ms':>10}{'tasks':>7}"
    print(header)
    ordered = sorted(
        results.items(),
        key=lambda item: item[1].get("cost_multiplier") if item[1].get("cost_multiplier") is not None else float("inf"),
    )
    for name, entry in ordered:
        times = [t["step_time_ms"] for t in entry.get("tasks", {}).values()]
        mult = entry.get("cost_multiplier")
        print(
            f"{name:<18}{fmt(mult):>7}"
            f"{fmt(sum(times) / len(times)):>10}{fmt(min(times)):>10}{fmt(max(times)):>10}"
            f"{len(times):>5}/7"
        )
    failed = cost.get("failed", {})
    for key, reason in failed.items():
        print(f"FAILED {key}: {reason.splitlines()[0][:80]}")


def _top_params(entry, limit=3):
    st = entry.get("sensitivity_ST") or {}
    pairs = [(k, v) for k, v in st.items() if isinstance(v, (int, float))]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return " ".join(f"{k}:{pct(v)}" for k, v in pairs[:limit])


def space_overview(cost, space):
    entries = space.get("optimizers", {})
    missing = [n for n in ALL_OPTIMIZERS if n not in entries]
    failed = [n for n, e in entries.items() if e.get("status") != "ok"]
    print(f"\n=== SEARCH SPACE ({len(entries) - len(failed)} ok, {len(failed)} failed, {len(missing)} missing) ===")
    if missing:
        print("missing:", ", ".join(missing))
    if failed:
        print("failed:", ", ".join(failed))

    mults = {n: e.get("cost_multiplier") for n, e in (cost.get("results") or {}).items()}
    header = (
        f"{'optimizer':<18}{'act':>4}{'fix':>4}{'conv':>6}{'sur':>4}"
        f"{'elapsed':>9}{'mult':>7}  top params (ST)"
    )
    print(header)
    for name in ALL_OPTIMIZERS:
        entry = entries.get(name)
        if not isinstance(entry, dict):
            continue
        active = len(entry.get("active_bounds") or {})
        fixed = len(entry.get("fixed") or {})
        conv = pct(entry.get("convergence_rate"))
        sur = "yes" if entry.get("surrogate_used") else "no"
        skipped = "*" if entry.get("calibration_skipped") else " "
        elapsed = entry.get("elapsed_seconds")
        elapsed_text = f"{elapsed / 60:.0f}m" if elapsed else "-"
        print(
            f"{name:<18}{active:>3}{skipped}{fixed:>4}{conv:>6}{sur:>4}"
            f"{elapsed_text:>9}{fmt(mults.get(name)):>7}  {_top_params(entry)}"
        )
    print("\n(*) universal policy instead of synthetic calibration")


def details(cost, space, name):
    cost_entry = (cost.get("results") or {}).get(name)
    space_entry = (space.get("optimizers") or {}).get(name)
    if cost_entry is None and space_entry is None:
        return False

    print(f"\n=== {name} ===")
    if cost_entry:
        mult = cost_entry.get("cost_multiplier")
        print(f"cost: mult={fmt(mult)} vs {cost.get('baseline')}")
        for task, timing in cost_entry.get("tasks", {}).items():
            print(f"  {task:<15}{fmt(timing['step_time_ms']):>10} ms +- {fmt(timing['step_time_std_ms'])}")

    if space_entry:
        elapsed = space_entry.get("elapsed_seconds")
        elapsed_text = f"{elapsed / 60:.0f}m" if elapsed and elapsed >= 120 else fmt(elapsed) + "s"
        print(f"status: {space_entry.get('status')}", end="")
        if space_entry.get("calibration_skipped"):
            print(f" (universal policy)", end="")
        print(f" | conv={pct(space_entry.get('convergence_rate'))}"
              f" | surrogate={'yes r2=' + fmt(space_entry.get('surrogate_r2')) if space_entry.get('surrogate_used') else 'no'}"
              f" | elapsed={elapsed_text}")

        st = space_entry.get("sensitivity_ST") or {}
        s1 = space_entry.get("sensitivity_S1") or {}
        bounds = space_entry.get("all_bounds") or {}
        active = set(space_entry.get("active_bounds") or {})
        fixed = space_entry.get("fixed") or {}

        rows = sorted(st.items(), key=lambda item: item[1] if isinstance(item[1], (int, float)) else -1, reverse=True)
        for param, st_value in rows:
            mark = "*" if param in active else " "
            lo, hi = (bounds.get(param) or [None, None])[:2]
            print(f" {mark} {param:<22}{fmt(lo):>12} .. {fmt(hi):<12} ST={pct(st_value):>5} S1={pct(s1.get(param))}")
        if fixed:
            print(" fixed:", ", ".join(f"{k}={fmt(v)}" for k, v in fixed.items()))
        if space_entry.get("skip_reason"):
            print(" note:", space_entry["skip_reason"])
    else:
        print("(no search space entry)")
    return True


def main():
    cost = load("cost.json") or {}
    space = load("search_space.json") or {}
    args = sys.argv[1:]

    queries = ALL_OPTIMIZERS if "--all" in args else [a for a in args if not a.startswith("-")]
    if not queries:
        cost_overview(cost)
        space_overview(cost, space)
        return

    found = False
    for query in queries:
        matches = [n for n in ALL_OPTIMIZERS if query in n] or [query]
        for name in matches:
            found |= details(cost, space, name)
    if not found:
        print("nothing found for:", ", ".join(queries))


if __name__ == "__main__":
    main()
