import inspect
import functools
import torch
import pytorch_optimizer as po


OPTIMIZER_TABLE = {
    "sgd": {"candidates": ["SGD"]},
    "nag": {"candidates": ["SGD"], "extra": {"nesterov": True, "momentum": 0.9, "dampening": 0.0}},
    "rmsprop": {"candidates": ["RMSProp", "RMSprop"]},
    "adadelta": {"candidates": ["AdaDelta", "Adadelta"]},
    "adam": {"candidates": ["Adam"]},
    "adamw": {"candidates": ["AdamW"]},
    "amsgrad": {"candidates": ["Adam"], "extra": {"amsgrad": True}},
    "nadam": {"candidates": ["NAdam"]},
    "radam": {"candidates": ["RAdam"]},
    "adabelief": {"candidates": ["AdaBelief"]},
    "diffgrad": {"candidates": ["DiffGrad"]},
    "yogi": {"candidates": ["Yogi"]},
    "lamb": {"candidates": ["Lamb"]},
    "adan": {"candidates": ["Adan"]},
    "lion": {"candidates": ["Lion"]},
    "amos": {"candidates": ["Amos"]},
    "prodigy": {"candidates": ["Prodigy"]},
    "schedulefreeadamw": {"candidates": ["ScheduleFreeAdamW"]},
    "ademamix": {"candidates": ["AdEMAMix"]},
    "adopt": {"candidates": ["ADOPT"]},
    "mars": {"candidates": ["MARS"]},
    "shampoo": {"candidates": ["Shampoo"]},
    "soap": {"candidates": ["SOAP"]},
    "muon": {"candidates": ["Muon"]},
    "lars": {"candidates": ["LARS"]},
    "madgrad": {"candidates": ["MADGRAD"]},
    "sm3": {"candidates": ["SM3"]},
}

SPECIAL_OPTIMIZERS = {"adammini", "lookaheadradam", "ranger21"}

NUMERIC_UNIT_HINTS = ("beta", "momentum", "rho", "alpha")

LOOKAHEAD_WRAPPER_KEYS = {"k", "alpha", "pullback_momentum"}

_RESOLVED = {}
_SOURCE = {}


def _resolve_class(candidates):
    for candidate in candidates:
        found = getattr(po, candidate, None)
        if found is not None:
            return found
    raise RuntimeError(f"none of {candidates} found in pytorch_optimizer")


def _init_table():
    for name, entry in OPTIMIZER_TABLE.items():
        cls = _resolve_class(entry["candidates"])
        extra = entry.get("extra", {})
        _RESOLVED[name] = functools.partial(cls, **extra) if extra else cls
        _SOURCE[name] = "pytorch_optimizer"


_init_table()


def full_registry():
    return list(OPTIMIZER_TABLE) + list(SPECIAL_OPTIMIZERS)


def optimizer_source(name):
    if name in _SOURCE:
        return _SOURCE[name]
    if name in SPECIAL_OPTIMIZERS:
        return "pytorch_optimizer(special)"
    raise KeyError(name)


def resolve_base(name):
    if name in _RESOLVED:
        return _RESOLVED[name]
    raise KeyError(name)


def _classify(pname):
    lowered = pname.lower()
    if "decay" in lowered:
        return "decay"
    if any(h in lowered for h in NUMERIC_UNIT_HINTS):
        return "unit"
    return "log"


def _entry(default, kind):
    if kind in ("log", "decay"):
        anchor = default if abs(default) > 1e-10 else 1e-4
    else:
        anchor = default
    return default, kind, anchor


def _extract_from_signature(sig, exclude):
    spec = {}
    for pname, param in sig.parameters.items():
        if pname in exclude or param.default is inspect.Parameter.empty:
            continue
        default = param.default
        if isinstance(default, bool) or default is None:
            continue
        if isinstance(default, tuple):
            for i, v in enumerate(default):
                if isinstance(v, (int, float)):
                    spec[f"{pname}_{i}"] = _entry(float(v), _classify(pname))
            continue
        if isinstance(default, (int, float)):
            spec[pname] = _entry(float(default), _classify(pname))
    return spec


def param_spec(name):
    exclude = {"self", "params", "defaults"}
    if name in _RESOLVED:
        cls = _RESOLVED[name]
        target = cls.func if isinstance(cls, functools.partial) else cls
        fixed = set(cls.keywords) if isinstance(cls, functools.partial) else set()
        return _extract_from_signature(inspect.signature(target.__init__), exclude | fixed)
    if name == "adammini":
        return _extract_from_signature(inspect.signature(po.AdamMini.__init__), exclude)
    if name == "lookaheadradam":
        base_spec = _extract_from_signature(inspect.signature(po.RAdam.__init__), exclude)
        wrapper_spec = _extract_from_signature(inspect.signature(po.Lookahead.__init__), exclude | {"optimizer"})
        return {**base_spec, **wrapper_spec}
    if name == "ranger21":
        return _extract_from_signature(inspect.signature(po.Ranger21.__init__), exclude | {"num_iterations"})
    raise KeyError(name)


def expand_flat_params(flat_point):
    kwargs = {}
    tuples = {}
    for name, value in flat_point.items():
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            tuples.setdefault(parts[0], {})[int(parts[1])] = value
        else:
            kwargs[name] = value
    for tname, parts in tuples.items():
        kwargs[tname] = tuple(parts[i] for i in sorted(parts))
    return kwargs


def default_kwargs(spec):
    flat = {name: value[0] for name, value in spec.items()}
    return expand_flat_params(flat)


def _wrap_as_module(named_params):
    module = torch.nn.Module()
    for i, (pname, p) in enumerate(named_params):
        module.register_parameter(f"p{i}_{pname}".replace(".", "_"), p)
    return module


def build_optimizer(name, params, kwargs, model=None, num_iterations=None):
    params = list(params)

    if name == "ranger21":
        if num_iterations is None:
            raise ValueError("ranger21 requires num_iterations")
        return po.Ranger21(params, num_iterations=num_iterations, **kwargs)

    if name == "adammini":
        target = model if model is not None else _wrap_as_module([(f"p{i}", p) for i, p in enumerate(params)])
        return po.AdamMini(target, **kwargs)

    if name == "lookaheadradam":
        wrapper_kwargs = {k: v for k, v in kwargs.items() if k in LOOKAHEAD_WRAPPER_KEYS}
        base_kwargs = {k: v for k, v in kwargs.items() if k not in LOOKAHEAD_WRAPPER_KEYS}
        base = po.RAdam(params, **base_kwargs)
        return po.Lookahead(base, params=params, **wrapper_kwargs)

    return resolve_base(name)(params, **kwargs)