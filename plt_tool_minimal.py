import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
from matplotlib.pyplot import figure
import json
import yaml
import os


def plot_(lbl: str, res_file: str, line_width=4, marker=None, line_style=None, color=None):
    with open(res_file, 'rb') as f:
        result = json.load(f)

    mean = result["mean_runs"]
    std = result["std_runs"]

    # UB = mean + 1 * std
    # LB = mean - 1 * std
    x = np.arange(len(mean))
    print('T = {}'.format(len(mean)))
    plt.plot(x, mean, label=lbl, linewidth=line_width, marker=marker, linestyle=line_style, color=color)
    # plt.fill_between(x, LB, UB, alpha=0.3, linewidth=0.5, color=color)


if __name__ == '__main__':
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # activate latex text rendering
    rc('text', usetex=True)
    root = os.getcwd()
    cfg = root + '/../plt_cfg.yaml'
    plt_cfg = yaml.load(open(cfg), Loader=yaml.FullLoader)

    ylim_b = plt_cfg["ylim_b"]
    ylim_t = plt_cfg["ylim_t"]
    xlim_l = plt_cfg["xlim_l"]
    xlim_r = plt_cfg["xlim_r"]

    for pl in plt_cfg["plots"]:
        result_file = pl["file"]
        lbl = pl["label"]
        lw = pl['line_width']
        ls = pl["line_style"]
        mk = pl["marker"]
        clr = pl["clr"]

        plot_(lbl=lbl,
              res_file=result_file,
              line_width=lw,
              marker=mk,
              line_style=ls,
              color=clr)

    plt.xlabel(r'Time Horizon', fontsize=10)
    plt.ylabel('Cumulative Regret', fontsize=10)

    plt.xlim(xlim_l, xlim_r)
    plt.ylim(ylim_b, ylim_t)

    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tick_params(labelsize=10)

    figure(figsize=(1, 1))
    plt.show()
