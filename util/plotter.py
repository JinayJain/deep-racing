import numpy as np
import visdom

vis = visdom.Visdom()


class Plotter:
    """
    Plot a visdom line chart
    """

    def __init__(self, title, xlabel, ylabel, update_interval=1, xmin=0):

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.update_interval = update_interval

        self.opts = {
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "xtickmin": xmin,
        }

        self.win = vis.line(
            X=np.array([0]), Y=np.array([0]), opts=self.opts, win=self.title
        )

        self.lines = {}

    def append(self, x, y, name):
        if name not in self.lines:
            self.lines[name] = [(x, y)]
        else:
            self.lines[name].append((x, y))

        if len(self.lines[name]) > self.update_interval:
            x, y = zip(*self.lines[name])

            vis.line(
                X=np.array(x),
                Y=np.array(y),
                win=self.win,
                name=name,
                update="append",
            )

            self.lines[name] = []

    def reset(self):
        for line in self.lines:
            vis.line(
                X=np.array([0]),
                Y=np.array([0]),
                win=self.win,
                name=line,
                update="replace"
            )

        self.lines = {}
