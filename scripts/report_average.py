import csv
import sys
from rich import print


def main():
    if len(sys.argv) != 2:
        print("Usage: {} <csv_file>".format(sys.argv[0]))
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, "r") as f:
        reader = csv.reader(f)
        total = 0
        num_items = 0

        next(reader)
        next(reader)

        for row in reader:
            total += float(row[1])
            num_items += 1

        print(f"Average reward: {total / num_items:.2f}")


if __name__ == "__main__":
    main()

