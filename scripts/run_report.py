import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, save_json
from optimbench.metrics import performance_profile, performance_ratios


TARGET_INDEX = 1
TAU_MAX = 32.0
TAUS = np.geomspace(1.0, TAU_MAX, 64)


def load_runs(results_path):
    data = json.loads(Path(results_path).read_text())
    runs_steps = {}
    runs_nfe = {}
    for task_name, payload in data.items():
        for optimizer_name, budgets in payload.get("optimizers", {}).items():
            for budget, result in budgets.items():
                key = f"{task_name}|{budget}"
                steps_to_targets = result.get("steps_to_targets") or []
                if len(steps_to_targets) > TARGET_INDEX and steps_to_targets[TARGET_INDEX]["mean"] is not None:
                    runs_steps.setdefault(key, {})[optimizer_name] = steps_to_targets[TARGET_INDEX]["mean"]
                    runs_nfe.setdefault(key, {})[optimizer_name] = result["total_nfe"]["mean"]
    return runs_steps, runs_nfe


def build_table(runs):
    return {key: solvers for key, solvers in runs.items() if len(solvers) >= 2}


def compute_profiles(table):
    ratios = performance_ratios(table)
    finite = [v for per_task in ratios.values() for v in per_task.values() if v is not None]
    r_m = (max(finite) * 2.0) if finite else TAU_MAX
    profiles = performance_profile(ratios, TAUS, r_m)
    return ratios, profiles, r_m


def plot_profiles(profiles, output_path, log2=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for solver in sorted(profiles):
        ax.plot(TAUS, profiles[solver], label=solver, marker="", linewidth=1.4)
    if log2:
        ax.set_xscale("log", base=2)
    ax.set_xlabel("tau" + (" (log2)" if log2 else ""))
    ax.set_ylabel("rho(tau)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def report(metric_table, metric_name, output_dir):
    table = build_table(metric_table)
    if not table:
        print(f"{metric_name}: not enough data")
        return None
    ratios, profiles, r_m = compute_profiles(table)
    plot_profiles(profiles, output_dir / f"profile_{metric_name}_linear.png")
    plot_profiles(profiles, output_dir / f"profile_{metric_name}_log2.png", log2=True)
    payload = {
        "metric": metric_name,
        "virtual_tasks": sorted(table),
        "r_m": r_m,
        "taus": TAUS.tolist(),
        "ratios": ratios,
        "profiles": profiles,
    }
    save_json(output_dir / f"report_{metric_name}.json", payload)

    print(f"\n{metric_name}: {len(table)} virtual tasks, {len(profiles)} optimizers (r_m={r_m:.2f})")
    print(f"{'optimizer':<24}{'rho(2)':>8}{'rho(4)':>8}")
    for solver in sorted(profiles, key=lambda s: -profiles[s][np.searchsorted(TAUS, 4.0)]):
        rho_2 = profiles[solver][np.searchsorted(TAUS, 2.0)]
        rho_4 = profiles[solver][np.searchsorted(TAUS, 4.0)]
        print(f"{solver:<24}{rho_2:>8.2f}{rho_4:>8.2f}")
    return payload


def main(results_path="results/benchmark_results.json", output_dir=None):
    results_path = Path(results_path)
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    output_dir = Path(output_dir) if output_dir else results_path.parent / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"no results file: {results_path}")
        return None

    runs_steps, runs_nfe = load_runs(results_path)
    steps_payload = report(runs_steps, "steps_to_target2", output_dir)
    nfe_payload = report(runs_nfe, "total_nfe", output_dir)
    print(f"\nartifacts saved to {output_dir}")
    return {"steps": steps_payload, "nfe": nfe_payload}


if __name__ == "__main__":
    main()
