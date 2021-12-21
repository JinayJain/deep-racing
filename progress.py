from rich.console import Group
from rich.live import Live
from rich.progress import Progress, BarColumn, TimeRemainingColumn
import time


def main():
    progress = Progress()
    task = progress.add_task("Task 1", total=100)
    progress_group = Group(progress)

    with Live(progress_group):
        for i in range(100):
            progress.advance(task)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
