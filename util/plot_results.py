import csv
import matplotlib.pyplot as plt


def main():
    # read CSV of episode results
    with open("episode_summaries.csv", "r") as f:
        reader = csv.reader(f)
        episodes = [(int(row[0]), float(row[1])) for row in reader]

    moving_avg = []
    for i in range(len(episodes)):
        if i == 0:
            moving_avg.append(episodes[i][1])
        else:
            moving_avg.append(0.95 * moving_avg[-1] + 0.05 * episodes[i][1])

    # plot results
    plt.style.use("fivethirtyeight")
    plt.title("Episode Reward over Time")
    plt.plot(
        [e[0] for e in episodes],
        [e[1] for e in episodes],
        label="Episode Rewards",
        color="blue",
        alpha=0.2,
    )
    plt.plot([e[0] for e in episodes], moving_avg, label="Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # set plot size
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)

    # save plot
    plt.savefig("episode_summaries.png", dpi=300)


if __name__ == "__main__":
    main()
