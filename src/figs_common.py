import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#000000", "#1F3A93", "#B22222", "#1B5E20", "#4D4D4D", "#5B3A29", "#B22222", "#1F3A93"]  # dark academic palette
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARK = ["o", "s", "^", "D", "v", "P"]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 10,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 7, "legend.frameon": False,
        "savefig.dpi": 400, "savefig.bbox": "tight", "figure.dpi": 100,
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
