from optimbench.smoke_test import run_smoke_test

if __name__ == "__main__":
    working, failing = run_smoke_test()
    print("working:", working)
    print("failing:")
    for name, error in failing.items():
        print(f"  {name}: {error}")
