#!/usr/bin/env python

"""plotTEC.py: module to plot tec data."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])
from matplotlib.dates import DateFormatter

def plot_timeseries(T, Ys, xlab="Time [UT]", fname="", txt=""):
    fig = plt.figure(dpi=240, figsize=(3,3))
    ax = fig.add_subplot(211)
    ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    ax.text(0.01, 1.05, txt, ha="left", va="center", transform=ax.transAxes, fontsize="x-small")
    Ysl, Ysr = Ys["left"], Ys["right"]
    for k in Ysl["entry"].keys():
        Y = Ysl["entry"][k]
        ax.plot(T, Y["v"], Y["color"]+Y["mk"], ls="None", ms=Y["ms"], alpha=Y["alpha"], label=Y["label"])
    ax.set_xlim(T[0], T[-1])
    ax.set_ylim(Ysl["ylim"])
    ax.set_ylabel(Ysl["ylabel"])
    ax.legend(loc=2, fontsize="xx-small", numpoints=3)
    ax = fig.add_subplot(212)
    ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    for k in Ysr["entry"].keys():
        Y = Ysr["entry"][k]
        ax.plot(T, Y["v"], Y["color"]+Y["mk"], ls="None", ms=Y["ms"], alpha=Y["alpha"], label=Y["label"])
    ax.set_xlim(T[0], T[-1])
    ax.set_ylim(Ysr["ylim"])
    ax.set_xlabel(xlab)
    ax.set_ylabel(Ysr["ylabel"])
    ax.legend(loc=1, fontsize="xx-small", numpoints=3)
    fig.savefig(fname, bbox_inches="tight")
    return