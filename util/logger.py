import csv
from rich.console import Console

console = Console()


class CSVLogger:
    def __init__(self, filename):
        self.filename = filename

        self.csv_file = open(self.filename, "w")
        self.csv_writer = csv.writer(self.csv_file)

    def write(self, data):
        self.csv_writer.writerow(data)
        self.csv_file.flush()

    def write_header(self, header):
        self.csv_writer.writerow(header)

    def close(self):
        self.csv_file.close()

