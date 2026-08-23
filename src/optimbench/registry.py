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


def _signature_target(name):
    if name == "adammini":
        return po.AdamMini
    if name == "lookaheadradam":
        return po.RAdam
    if name == "ranger21":
        return po.Ranger21
    if name == "muon":
        return po.Muon
    cls = _RESOLVED.get(name)
    if cls is None:
        return None
    return cls.func if isinstance(cls, functools.partial) else cls


def coerce_kwargs(target, kwargs):
    """Cast sampled/stored hyperparams back to the constructor's expected types."""
    try:
        sig = inspect.signature(target.__init__)
    except (TypeError, ValueError):
        return dict(kwargs)
    coerced = dict(kwargs)
    for pname, param in sig.parameters.items():
        if pname not in coerced:
            continue
        default = param.default
        value = coerced[pname]
        if isinstance(default, bool) or isinstance(value, bool):
            continue
        if isinstance(default, int):
            coerced[pname] = int(round(float(value)))
        elif isinstance(default, tuple) and isinstance(value, (list, tuple)):
            coerced[pname] = tuple(
                int(round(float(v))) if isinstance(d, int) and not isinstance(d, bool) else float(v)
                for d, v in zip(default, value)
            )
        elif isinstance(default, float):
            coerced[pname] = float(value)
    return coerced


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


class _PatchedAdamMini(po.AdamMini):
    @staticmethod
    def step_attn_proj(
        p,
        grad,
        state,
        parameter_per_head: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(p, dtype=torch.float32).view(-1, parameter_per_head)
            state["head"] = state["m"].shape[0]
            state["v_mean"] = torch.zeros(state["head"], device=state["m"].device)

        m, v = state["m"], state["v_mean"]

        head: int = state["head"]
        grad = grad.view(head, parameter_per_head)

        m.lerp_(grad, weight=1.0 - beta1)

        tmp_lr = torch.mean(grad * grad, dim=1).to(m.device)
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        update = (1 / (h * bias_correction1)).view(head, 1).mul(m)

        if p.dim() > 1:
            d0, d1 = p.size()
            update = update.view(d0, d1)
        else:
            update = update.view(-1)

        p.add_(update, alpha=-lr)

    @staticmethod
    def step_attn(
        p,
        grad,
        state,
        num_heads: int,
        q_per_kv: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(p, dtype=torch.float32).view(num_heads, q_per_kv + 2, -1)
            state["v_mean"] = torch.zeros(num_heads, q_per_kv + 2, device=state["m"].device)

        m, v = state["m"], state["v_mean"]

        grad = grad.view(num_heads, q_per_kv + 2, -1)

        m.lerp_(grad, weight=1.0 - beta1)

        tmp_lr = torch.mean(grad * grad, dim=2).to(m.device)
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        update = (1 / (h * bias_correction1)).unsqueeze(-1).mul(m)

        if p.dim() > 1:
            d0, d1 = p.size()
            update = update.view(d0, d1)
        else:
            update = update.view(-1)

        p.add_(update, alpha=-lr)


def build_optimizer(name, params, kwargs, model=None, num_iterations=None):
    params = list(params)
    kwargs = dict(kwargs)

    if name == "ranger21":
        if num_iterations is None:
            raise ValueError("ranger21 requires num_iterations")
        kwargs = coerce_kwargs(po.Ranger21, kwargs)
        return po.Ranger21(params, num_iterations=num_iterations, **kwargs)

    if name == "adammini":
        kwargs = coerce_kwargs(po.AdamMini, kwargs)
        target = model if model is not None else _wrap_as_module([(f"p{i}", p) for i, p in enumerate(params)])
        return _PatchedAdamMini(target, **kwargs)

    if name == "lookaheadradam":
        wrapper_kwargs = {k: v for k, v in kwargs.items() if k in LOOKAHEAD_WRAPPER_KEYS}
        base_kwargs = {k: v for k, v in kwargs.items() if k not in LOOKAHEAD_WRAPPER_KEYS}
        wrapper_kwargs = coerce_kwargs(po.Lookahead, wrapper_kwargs)
        base_kwargs = coerce_kwargs(po.RAdam, base_kwargs)
        base = po.RAdam(params, **base_kwargs)
        return po.Lookahead(base, params=params, **wrapper_kwargs)

    if name == "muon":
        kwargs = coerce_kwargs(po.Muon, kwargs)
        matrix_params = [p for p in params if p.ndim >= 2]
        vector_params = [p for p in params if p.ndim < 2]
        groups = []
        if matrix_params:
            groups.append({"params": matrix_params, "use_muon": True})
        if vector_params:
            groups.append({"params": vector_params, "use_muon": False})
        ns_steps = kwargs.pop("ns_steps", 5)
        return po.Muon(groups, ns_steps=ns_steps, **kwargs)

    cls = resolve_base(name)
    kwargs = coerce_kwargs(_signature_target(name) or cls, kwargs)
    return cls(params, **kwargs)