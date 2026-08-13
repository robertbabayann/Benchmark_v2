from optimbench.pipeline import run_pair
from optimbench.tasks.pinn import BurgersTask

if __name__ == "__main__":
    task = BurgersTask()
    results, targets = run_pair(task, "adamw")
    print(targets)
    for budget, data in results.items():
        print(budget, data)
