import numpy as np
import visdom

vis = visdom.Visdom()


class Plotter:
    """
    Plot a visdom line chart
    """

    def __init__(self, title, xlabel, ylabel, update_interval=1):

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.opts = {
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
        }

        self.win = vis.line(
            X=np.array([0]), Y=np.array([0]), opts=self.opts, win=self.title
        )

        self.lines = set()

    def append(self, x, y, name=None):
        if name is not None:
            self.lines.add(name)

        vis.line(
            X=np.array([x]), Y=np.array([y]), win=self.win, update="append", name=name
        )

    def clear(self):
        for line in self.lines:
            vis.line(
                X=np.array([0]),
                Y=np.array([0]),
                win=self.win,
                name=line,
                opts=self.opts,
            )

        self.lines = set()

