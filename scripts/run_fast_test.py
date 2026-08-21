from optimbench.fast_test import run_fast_test

from common import save_json, select_optimizers
from settings import FAST_TEST


def main():
    names = select_optimizers(FAST_TEST.optimizers)
    print(f"fast test: {len(names)} optimizers, {FAST_TEST.steps} steps, seed {FAST_TEST.seed}")
    working, failing = run_fast_test(names=names, steps=FAST_TEST.steps, seed=FAST_TEST.seed)

    print(f"\nworking ({len(working)}):")
    for name in working:
        print(f"  {name}")
    print(f"\nfailing ({len(failing)}):")
    for name, error in failing.items():
        print(f"  {name}: {error}")

    if FAST_TEST.report_path:
        save_json(
            FAST_TEST.report_path,
            {"steps": FAST_TEST.steps, "seed": FAST_TEST.seed, "working": working, "failing": failing},
        )
        print(f"\nreport saved to {FAST_TEST.report_path}")
    return working, failing


if __name__ == "__main__":
    main()
