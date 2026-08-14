from optimbench.smoke_test import run_smoke_test
from optimbench.registry import optimizer_source

if __name__ == "__main__":
    working, failing = run_smoke_test()
    print("working:")
    for name in working:
        print(f"  {name}: {optimizer_source(name)}")
    print("failing:")
    for name, error in failing.items():
        print(f"  {name}: {error}")