import time
import traceback
import dataclasses
from datetime import datetime, timezone
from tqdm import tqdm

from optimbench.config import CALIBRATION as DEFAULT_CALIBRATION
from optimbench.registry import param_spec, optimizer_source
from optimbench.calibration import calibrate

from common import save_json, select_optimizers
from settings import CALIBRATION_RUN


def convergence_rate(records):
    if not records:
        return None
    converged = sum(1 for r in records if r["status"] == "converged")
    return converged / len(records)


def serialize_bounds(bounds):
    return {k: [v[0], v[1]] for k, v in bounds.items()}


def build_config():
    return dataclasses.replace(
        DEFAULT_CALIBRATION,
        sobol_points=CALIBRATION_RUN.sobol_points,
        seeds_per_point=CALIBRATION_RUN.seeds_per_point,
        steps_per_run=CALIBRATION_RUN.steps_per_run,
        sensitivity_samples=CALIBRATION_RUN.sensitivity_samples,
        bound_low_percentile=CALIBRATION_RUN.bound_low_percentile,
        bound_high_percentile=CALIBRATION_RUN.bound_high_percentile,
        use_surrogate=CALIBRATION_RUN.use_surrogate,
    )


def calibrate_all(cfg, names, output_path):
    summary = {"config": dataclasses.asdict(cfg), "optimizers": {}}
    progress = tqdm(names, desc="calibrating", unit="optimizer")
    for name in progress:
        progress.set_postfix_str(name)
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        try:
            spec = param_spec(name)
            result = calibrate(name, spec, cfg)
            entry = {
                "status": "ok",
                "source": optimizer_source(name),
                "calibration_skipped": result["calibration_skipped"],
                "skip_reason": result["skip_reason"],
                "active_bounds": serialize_bounds(result["active_bounds"]),
                "all_bounds": serialize_bounds(result["all_bounds"]),
                "fixed": result["fixed"],
                "sensitivity_ST": dict(zip(result["sensitivity"]["names"], result["sensitivity"]["ST"])),
                "sensitivity_S1": dict(zip(result["sensitivity"]["names"], result["sensitivity"]["S1"])),
                "surrogate_used": result["surrogate_used"],
                "surrogate_r2": result["surrogate_r2"],
                "convergence_rate": convergence_rate(result["records"]),
            }
        except Exception as e:
            entry = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
        entry["started_at"] = started_at
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        entry["elapsed_seconds"] = time.perf_counter() - start
        summary["optimizers"][name] = entry
        save_json(output_path, summary)
    return summary


def main():
    cfg = build_config()
    names = select_optimizers(CALIBRATION_RUN.optimizers)
    print(f"calibration: {len(names)} optimizers -> {CALIBRATION_RUN.output_path}")
    summary = calibrate_all(cfg, names, CALIBRATION_RUN.output_path)
    failed = [name for name, entry in summary["optimizers"].items() if entry["status"] != "ok"]
    print(f"done: {len(names) - len(failed)} ok, {len(failed)} failed")
    return summary


if __name__ == "__main__":
    main()
