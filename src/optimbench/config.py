from dataclasses import dataclass, field


@dataclass
class CalibrationConfig:
    sobol_points: int = 512
    seeds_per_point: int = 3
    steps_per_run: int = 300
    log_scale_span: float = 4.0
    unit_eps: float = 1e-3
    divergence_threshold: float = 1e6
    sensitivity_samples: int = 64
    sensitivity_threshold: float = 0.05


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
