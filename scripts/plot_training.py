import csv
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: {} <csv_file>".format(sys.argv[0]))
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, "r") as f:
        reader = csv.reader(f)

        header = next(reader)
        print(header)


if __name__ == "__main__":
    main()
