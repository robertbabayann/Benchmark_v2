from dataclasses import dataclass, field


@dataclass
class CalibrationConfig:
    sobol_points: int = 256
    seeds_per_point: int = 3
    steps_per_run: int = 100
    log_scale_span: float = 4.0
    decay_log_scale_span: float = 2.0
    decay_hard_cap: float = 0.5
    unit_eps: float = 1e-3
    divergence_threshold: float = 1e6
    sensitivity_samples: int = 256
    sensitivity_threshold: float = 0.05
    bound_low_percentile: float = 15.0
    bound_high_percentile: float = 85.0
    log_clip_multiplier: float = 1000.0
    decay_clip_multiplier: float = 50.0
    use_surrogate: bool = False
    surrogate_min_r2: float = 0.85
    surrogate_min_samples: int = 50
    surrogate_cv_folds: int = 5
    surrogate_bounds_expansion: float = 1.3

@dataclass
class TargetConfig:
    levels: tuple = (0.5, 0.75, 0.9)
    max_budget_multiplier: float = 1.5
    calibration_seeds: int = 5


@dataclass
class TuningConfig:
    budgets: dict = field(default_factory=lambda: {"no": 0, "low": 20, "medium": 50, "high": 100})
    sampler_seed: int = 0


@dataclass
class RunConfig:
    final_seeds: int = 5


CALIBRATION = CalibrationConfig()
TARGETS = TargetConfig()
TUNING = TuningConfig()
RUN = RunConfig()