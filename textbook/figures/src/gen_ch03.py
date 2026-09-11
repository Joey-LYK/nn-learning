# -*- coding: utf-8 -*-
"""生成第 3 章插图：测量直方图、68-95-99.7 法则、标准差宽窄对比。"""

import matplotlib.pyplot as plt
import numpy as np

from fig_common import (
    ACCENT,
    FILL,
    INK,
    MUTED,
    PRIMARY,
    new_figure,
    save_figure,
    setup_style,
)


def _norm_pdf(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def plot_histogram():
    centers = np.arange(119.7, 120.31, 0.05)
    counts = np.array([3, 12, 45, 120, 240, 380, 480, 380, 240, 120, 45, 12, 3])

    fig, ax = new_figure()
    ax.bar(centers, counts, width=0.04, color=FILL, edgecolor=PRIMARY,
           linewidth=1.2, zorder=3)
    ax.annotate("均值 120", xy=(120.0, 500), xytext=(120.22, 440),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("测量值（厘米）")
    ax.set_ylabel("次数")
    ax.set_xlim(119.62, 120.38)
    ax.set_ylim(0, 560)
    ax.set_xticks([119.7, 120.0, 120.3])
    save_figure(fig, "ch03-histogram.svg")


def plot_normal_68():
    x = np.linspace(-4, 4, 500)
    y = _norm_pdf(x, 0, 1)

    fig, ax = new_figure()
    ax.plot(x, y, color=PRIMARY, linewidth=2.5, zorder=3)
    m2 = (x >= -2) & (x <= 2)
    ax.fill_between(x[m2], y[m2], color=PRIMARY, alpha=0.12, zorder=2)
    m1 = (x >= -1) & (x <= 1)
    ax.fill_between(x[m1], y[m1], color=PRIMARY, alpha=0.30, zorder=2)
    for k in (-1, 1):
        ax.axvline(k, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)

    ax.text(0, 0.13, "约 68%", ha="center", fontsize=12, color=INK)
    ax.annotate("约 95%", xy=(1.55, 0.09), xytext=(2.6, 0.2),
                fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))

    ax.set_xlabel("测量值")
    ax.set_ylabel("概率密度")
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(0, 0.46)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels([r"$\mu-2\sigma$", r"$\mu-\sigma$", r"$\mu$",
                        r"$\mu+\sigma$", r"$\mu+2\sigma$"])
    ax.set_yticks([])
    save_figure(fig, "ch03-normal-68.svg")


def plot_sigma_compare():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    cases = [(ax1, 5, "窄而高", (152, 188), [165, 170, 175]),
             (ax2, 15, "扁而宽", (115, 225), [155, 170, 185])]
    for ax, sigma, word, xlim, ticks in cases:
        x = np.linspace(170 - 4 * sigma, 170 + 4 * sigma, 400)
        y = _norm_pdf(x, 170, sigma)
        ax.plot(x, y, color=PRIMARY, linewidth=2.5, zorder=3)
        ax.fill_between(x, y, color=FILL, zorder=2)
        ax.set_title(f"$\\sigma={sigma}$（{word}）", fontsize=12, color=INK)
        ax.set_xlabel("身高（厘米）")
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks)
        ax.set_yticks([])
    ax1.set_ylabel("概率密度")
    save_figure(fig, "ch03-sigma-compare.svg")


if __name__ == "__main__":
    setup_style()
    plot_histogram()
    plot_normal_68()
    plot_sigma_compare()
