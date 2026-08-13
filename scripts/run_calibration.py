from optimbench.config import CALIBRATION
from optimbench.registry import param_spec
from optimbench.calibration import calibrate

if __name__ == "__main__":
    for name in ["sgd", "adam", "adamw"]:
        spec = param_spec(name)
        result = calibrate(name, spec, CALIBRATION)
        print(name, "bounds:", result["bounds"])
        print(name, "fixed:", result["fixed"])
        print(name, "S1:", dict(zip(result["sensitivity"]["names"], result["sensitivity"]["S1"])))
