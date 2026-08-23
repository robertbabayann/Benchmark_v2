from __future__ import annotations

import dataclasses
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from optimbench.calibration import calibrate
from optimbench.registry import optimizer_source

PRESETS_DIR = Path(__file__).resolve().parent / "presets"
SEARCH_SPACE_PATH = PRESETS_DIR / "search_space.json"
COST_PATH = PRESETS_DIR / "cost.json"


def sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def serialize_bounds(bounds):
    return {k: [v[0], v[1]] for k, v in bounds.items()}


def convergence_rate(records):
    if not records:
        return None
    converged = sum(1 for r in records if r["status"] == "converged")
    return converged / len(records)


def load_payload(path: str | Path = SEARCH_SPACE_PATH):
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        data = json.loads(file_path.read_text())
    except json.JSONDecodeError:
        print(f"warning: corrupted preset ignored: {file_path}")
        return None
    return data if isinstance(data, dict) else None


def load_ok_entries(path: str | Path = SEARCH_SPACE_PATH):
    data = load_payload(path)
    if not data:
        return {}
    optimizers = data.get("optimizers")
    if not isinstance(optimizers, dict):
        return {}
    return {
        name: entry
        for name, entry in optimizers.items()
        if isinstance(entry, dict) and entry.get("status") == "ok"
    }


def _bounds_pairs(raw):
    pairs = {}
    for name, value in raw.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            pairs[name] = (float(value[0]), float(value[1]))
    return pairs


def entry_to_result(entry):
    st = entry.get("sensitivity_ST") or {}
    s1 = entry.get("sensitivity_S1") or {}
    names = list(st.keys())
    return {
        "active_bounds": _bounds_pairs(entry.get("active_bounds") or {}),
        "all_bounds": _bounds_pairs(entry.get("all_bounds") or {}),
        "fixed": dict(entry.get("fixed") or {}),
        "sensitivity": {"names": names, "S1": [s1.get(n) for n in names], "ST": [st.get(n) for n in names]},
        "records": [],
        "surrogate_used": bool(entry.get("surrogate_used")),
        "surrogate_r2": entry.get("surrogate_r2"),
        "calibration_skipped": bool(entry.get("calibration_skipped")),
        "skip_reason": entry.get("skip_reason"),
        "from_preset": True,
    }


def build_entry(optimizer_name, result, started_at=None, elapsed_seconds=None):
    moment = datetime.now(timezone.utc).isoformat()
    names = result["sensitivity"]["names"]
    return {
        "status": "ok",
        "source": optimizer_source(optimizer_name),
        "calibration_skipped": result["calibration_skipped"],
        "skip_reason": result["skip_reason"],
        "active_bounds": serialize_bounds(result["active_bounds"]),
        "all_bounds": serialize_bounds(result["all_bounds"]),
        "fixed": result["fixed"],
        "sensitivity_ST": dict(zip(names, result["sensitivity"]["ST"])),
        "sensitivity_S1": dict(zip(names, result["sensitivity"]["S1"])),
        "surrogate_used": result["surrogate_used"],
        "surrogate_r2": result["surrogate_r2"],
        "convergence_rate": convergence_rate(result["records"]),
        "started_at": started_at or moment,
        "finished_at": moment,
        "elapsed_seconds": elapsed_seconds,
    }


def save_result(optimizer_name, result, path: str | Path = SEARCH_SPACE_PATH, cfg=None, elapsed_seconds=None):
    data = load_payload(path) or {}
    optimizers = data.get("optimizers")
    if not isinstance(optimizers, dict):
        optimizers = {}
    if cfg is not None and "config" not in data:
        data["config"] = dataclasses.asdict(cfg)
    data["optimizers"] = optimizers
    optimizers[optimizer_name] = build_entry(optimizer_name, result, elapsed_seconds=elapsed_seconds)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(sanitize(data), indent=2))
    return file_path


def _compatible(entry, param_spec):
    stored = set(entry.get("active_bounds") or {}) | set(entry.get("fixed") or {})
    return bool(stored) and stored <= set(param_spec)


def get_calibration(optimizer_name, param_spec, cfg, path: str | Path = SEARCH_SPACE_PATH):
    entry = load_ok_entries(path).get(optimizer_name)
    if entry is not None and _compatible(entry, param_spec):
        print(f"preset: '{optimizer_name}' loaded from {Path(path).name}")
        return entry_to_result(entry)
    if entry is not None:
        print(f"warning: preset for '{optimizer_name}' is outdated (parameters mismatch), recalibrating")
    else:
        print(
            f"warning: no search-space preset for '{optimizer_name}', "
            f"running automatic calibration now (this may take a long time)"
        )
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    result = calibrate(optimizer_name, param_spec, cfg)
    elapsed = time.perf_counter() - start
    save_result(optimizer_name, result, path=path, cfg=cfg, elapsed_seconds=elapsed)
    print(f"preset: '{optimizer_name}' calibrated in {elapsed:.0f}s, appended to {Path(path).name}")
    return result


def get_cost_multiplier(optimizer_name, task_name, path: str | Path = COST_PATH):
    data = load_payload(path)
    if not data:
        return None
    results = data.get("results") or {}
    baseline_name = data.get("baseline", "adamw")
    opt_tasks = (results.get(optimizer_name) or {}).get("tasks") or {}
    base_tasks = (results.get(baseline_name) or {}).get("tasks") or {}
    if task_name in opt_tasks and task_name in base_tasks:
        opt_ms = (opt_tasks[task_name] or {}).get("step_time_ms")
        base_ms = (base_tasks[task_name] or {}).get("step_time_ms")
        if opt_ms and base_ms:
            ratio = float(opt_ms) / float(base_ms)
            print(f"cost preset: '{optimizer_name}' on '{task_name}' mult={ratio:.3f} (from {Path(path).name})")
            return ratio
    return None
