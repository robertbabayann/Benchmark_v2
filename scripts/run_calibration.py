import time
import traceback
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from optimbench.config import CALIBRATION as DEFAULT_CALIBRATION
from optimbench.registry import param_spec
from optimbench.calibration import calibrate
from optimbench.presets import build_entry, load_payload

from common import ROOT, save_json, select_optimizers
from settings import CALIBRATION_RUN


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


def collect_pending(names, stored):
    done = [
        name for name in names
        if isinstance(stored.get(name), dict) and stored[name].get("status") == "ok"
    ]
    pending = [name for name in names if name not in done]
    return done, pending


def calibrate_all(cfg, names, output_path, summary):
    progress = tqdm(names, desc="calibrating", unit="optimizer")
    interrupted = False
    try:
        for name in progress:
            progress.set_postfix_str(name)
            started_at = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()
            try:
                spec = param_spec(name)
                result = calibrate(name, spec, cfg)
                entry = build_entry(
                    name, result,
                    started_at=started_at,
                    elapsed_seconds=time.perf_counter() - start,
                )
            except Exception as e:
                entry = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
            summary["optimizers"][name] = entry
            save_json(output_path, summary)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted, progress saved")
    return interrupted


def main():
    cfg = build_config()
    names = select_optimizers(CALIBRATION_RUN.optimizers)

    previous = load_payload(CALIBRATION_RUN.output_path)
    stored = {}
    if previous and isinstance(previous.get("optimizers"), dict):
        stored = previous["optimizers"]
    if previous is not None and not stored:
        print(f"warning: no usable optimizer entries found in {CALIBRATION_RUN.output_path}, starting fresh")

    summary = {"config": dataclasses.asdict(cfg), "optimizers": dict(stored)}
    done, pending = collect_pending(names, stored)
    print(
        f"calibration: {len(names)} optimizers selected, "
        f"{len(done)} already calibrated, {len(pending)} to do -> {CALIBRATION_RUN.output_path}"
    )
    if not pending:
        print("nothing to do: all selected optimizers are already calibrated")
        return summary

    interrupted = calibrate_all(cfg, pending, CALIBRATION_RUN.output_path, summary)
    failed = [n for n in pending if summary["optimizers"].get(n, {}).get("status") != "ok"]
    print(f"done: {len(pending) - len(failed)} newly calibrated ok, {len(failed)} failed")
    status = "partially complete (interrupted)" if interrupted else "done"
    print(f"{status}, results saved to {CALIBRATION_RUN.output_path}")
    return summary


if __name__ == "__main__":
    main()
