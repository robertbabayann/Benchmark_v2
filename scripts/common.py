import json
import math
from functools import partial
from importlib import import_module
from pathlib import Path

from optimbench.registry import full_registry


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

TASK_TABLE = {
    "burgers_pinn": ("optimbench.tasks.pinn", "BurgersTask", {}),
    "cifar100": ("optimbench.tasks.cv", "Cifar100Task", {"data_root": str(DATA_ROOT / "cifar100")}),
    "tiny_imagenet": ("optimbench.tasks.cv", "TinyImageNetTask", {"data_root": str(DATA_ROOT / "tiny-imagenet-200")}),
    "ogbg_molhiv": ("optimbench.tasks.gnn", "MolHivTask", {"data_root": str(DATA_ROOT / "molhiv")}),
    "chargpt": ("optimbench.tasks.nlp", "CharGPTTask", {"text_path": str(DATA_ROOT / "shakespeare" / "input.txt")}),
    "warpeace": ("optimbench.tasks.nlp", "WarPeaceTask", {"text_path": str(DATA_ROOT / "warpeace" / "input.txt")}),
    "movielens": ("optimbench.tasks.recsys", "MovieLensTask", {"ratings_path": str(DATA_ROOT / "ml-1m" / "ratings.dat")}),
}


def resolve_task(name):
    if name not in TASK_TABLE:
        raise KeyError(f"unknown task: {name}")
    module_name, class_name, kwargs = TASK_TABLE[name]
    return partial(getattr(import_module(module_name), class_name), **kwargs)


def select_tasks(names):
    chosen = list(names) if names else list(TASK_TABLE)
    for name in chosen:
        if name not in TASK_TABLE:
            raise KeyError(f"unknown task: {name} (available: {', '.join(TASK_TABLE)})")
    return chosen


def select_optimizers(names):
    return list(names) if names else full_registry()


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


def save_json(path, payload):
    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sanitize(payload), indent=2))
