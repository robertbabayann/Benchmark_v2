from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = ROOT / "src" / "optimbench" / "presets"


@dataclass
class FastTestSettings:
    steps: int = 5
    seed: int = 0
    optimizers: tuple = ()
    report_path: str = ""


@dataclass
class CalibrationSettings:
    output_path: str = str(PRESETS_DIR / "search_space.json")
    optimizers: tuple = ()
    sobol_points: int = 256
    seeds_per_point: int = 3
    steps_per_run: int = 100
    sensitivity_samples: int = 256
    bound_low_percentile: float = 15.0
    bound_high_percentile: float = 85.0
    use_surrogate: bool = False


@dataclass
class PipelineSettings:
    task: str = "burgers_pinn"
    device: str = "auto"
    optimizer: str = "adamw"
    reference_name: str = "adamw"
    reference_trials: Optional[int] = None
    budgets: dict = field(default_factory=lambda: {"no": 0, "low": 20, "medium": 50, "high": 100})
    final_seeds: int = 5
    sampler_seed: int = 0
    output_path: str = ""


@dataclass
class CostSettings:
    tasks: tuple = ()
    optimizers: tuple = ()
    device: str = "auto"
    baseline: str = "adamw"
    timing_steps: int = 50
    seeds: int = 10
    retry_failed: bool = False
    output_path: str = str(PRESETS_DIR / "cost.json")


@dataclass
class GoalSettings:
    task: str = "burgers_pinn"
    device: str = "auto"
    reference_name: str = "adamw"
    reference_trials: Optional[int] = None
    final_seeds: int = 5
    sampler_seed: int = 0
    output_path: str = str(PRESETS_DIR / "goals.json")


@dataclass
class BenchmarkSettings:
    tasks: tuple = ()
    optimizers: tuple = ()
    device: str = "auto"
    reference_name: str = "adamw"
    reference_trials: Optional[int] = None
    budgets: dict = field(default_factory=lambda: {"no": 0, "low": 20, "medium": 50, "high": 100})
    final_seeds: int = 5
    sampler_seed: int = 0
    output_path: str = "results/benchmark_results.json"


FAST_TEST = FastTestSettings()
CALIBRATION_RUN = CalibrationSettings()
GOAL = GoalSettings()
PIPELINE = PipelineSettings()
COST = CostSettings()
BENCHMARK = BenchmarkSettings()
