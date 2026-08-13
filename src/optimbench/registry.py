import inspect
import functools
import torch
import pytorch_optimizer as po


TORCH_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "nag": functools.partial(torch.optim.SGD, nesterov=True, momentum=0.9),
    "rmsprop": torch.optim.RMSprop,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "amsgrad": functools.partial(torch.optim.Adam, amsgrad=True),
    "adagrad": torch.optim.Adagrad,
}

PYTORCH_OPTIMIZER_NAMES = {
    "adabelief": "AdaBelief",
    "diffgrad": "DiffGrad",
    "yogi": "Yogi",
    "lamb": "Lamb",
    "adan": "Adan",
    "lion": "Lion",
    "adammini": "AdamMini",
    "amos": "Amos",
    "prodigy": "Prodigy",
    "schedulefreeadamw": "ScheduleFreeAdamW",
    "lookaheadradam": "Lookahead",
    "ranger21": "Ranger21",
    "ademamix": "AdEMAMix",
    "adopt": "ADOPT",
    "mars": "MARS",
    "shampoo": "Shampoo",
    "soap": "SOAP",
}

NUMERIC_UNIT_HINTS = ("beta", "momentum", "rho", "alpha", "decay_rate")


def resolve_optimizer(name):
    if name in TORCH_OPTIMIZERS:
        return TORCH_OPTIMIZERS[name]
    if name in PYTORCH_OPTIMIZER_NAMES:
        return getattr(po, PYTORCH_OPTIMIZER_NAMES[name])
    raise KeyError(name)


def full_registry():
    return list(TORCH_OPTIMIZERS) + list(PYTORCH_OPTIMIZER_NAMES)


def _classify(pname):
    lowered = pname.lower()
    if any(h in lowered for h in NUMERIC_UNIT_HINTS):
        return "unit"
    return "log"


def _entry(default, kind):
    if kind == "log":
        anchor = default if abs(default) > 1e-8 else 1e-4
    else:
        anchor = default
    return default, kind, anchor


def param_spec(name):
    cls = resolve_optimizer(name)
    target = cls.func if isinstance(cls, functools.partial) else cls
    sig = inspect.signature(target.__init__)
    fixed = set(cls.keywords) if isinstance(cls, functools.partial) else set()
    spec = {}
    for pname, param in sig.parameters.items():
        if pname in ("self", "params", "defaults") or pname in fixed:
            continue
        if param.default is inspect.Parameter.empty:
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


def default_kwargs(spec):
    return {name: value[0] for name, value in spec.items()}
