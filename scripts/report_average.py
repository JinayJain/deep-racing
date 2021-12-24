import csv
import sys
from rich import print
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Usage: {} <csv_file>".format(sys.argv[0]))
        sys.exit(1)

    data = []

    filename = sys.argv[1]
    with open(filename, "r") as f:
        reader = csv.reader(f)

        # skip header
        next(reader)

        for row in reader:
            data.append(float(row[1]))

    data = np.array(data)

    print("Mean:", np.mean(data))
    print("Standard Error:", np.std(data) / np.sqrt(len(data)))


if __name__ == "__main__":
    main()

