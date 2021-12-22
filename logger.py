import rich
from rich.console import Console

console = Console()


class Logger:
    def __init__(self):
        self.tracks = {}

    def log(self, name, value):
        if name not in self.tracks:
            self.tracks[name] = []
        self.tracks[name].append(value)

    def print(self, title=None):
        if title is not None:
            console.rule(title)

        for name, track in self.tracks.items():
            console.print(f"{name}: {track[-1]}")

