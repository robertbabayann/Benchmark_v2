import json
import math
import time
import traceback
import dataclasses
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

from optimbench.config import CALIBRATION
from optimbench.registry import full_registry, param_spec, optimizer_source
from optimbench.calibration import calibrate


def convergence_rate(records):
    if not records:
        return None
    converged = sum(1 for r in records if r["status"] == "converged")
    return converged / len(records)


def serialize_bounds(bounds):
    return {k: [v[0], v[1]] for k, v in bounds.items()}


def sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def run_full_calibration(output_path):
    names = full_registry()
    summary = {"config": dataclasses.asdict(CALIBRATION), "optimizers": {}}

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    progress = tqdm(names, desc="calibrating", unit="optimizer")
    for name in progress:
        progress.set_postfix_str(name)
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        try:
            spec = param_spec(name)
            result = calibrate(name, spec, CALIBRATION)
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
        output.write_text(json.dumps(sanitize(summary), indent=2))

    return summary


if __name__ == "__main__":
    run_full_calibration("results/calibration_summary.json")