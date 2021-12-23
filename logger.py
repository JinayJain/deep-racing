import csv
from rich.console import Console

console = Console()


class Logger:
    def __init__(self, logfile):
        self.tracks = {}

        self.file = open(logfile, "w")
        self.writer = csv.writer(self.file)

    def close(self):
        self.file.close()

    def log(self, name, value):
        if name not in self.tracks:
            self.tracks[name] = []
        self.tracks[name].append(value)

    def print(self, title=None):
        if title is not None:
            console.rule(title)

        for name, track in self.tracks.items():
            console.print(f"{name}: {track[-1]}")

    def write(self):
        if self.file.tell() == 0:
            self.writer.writerow(list(self.tracks.keys()))

        row = [track[-1] for track in self.tracks.values()]
        self.writer.writerow(row)
        self.file.flush()

