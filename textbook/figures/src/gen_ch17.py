# -*- coding: utf-8 -*-
"""生成第 17 章插图：圆内圆外分类任务与学到的决策边界。"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from fig_common import ACCENT, FILL, INK, MUTED, PRIMARY, new_figure, save_figure, setup_style


def plot_circle_boundary():
    grid = np.linspace(-1.05, 1.05, 300)
    x1, x2 = np.meshgrid(grid, grid)
    r2 = x1 ** 2 + x2 ** 2
    prob = 1.0 / (1.0 + np.exp(-12.0 * (0.25 - r2)))

    rng = np.random.default_rng(7)
    n_in, n_out = 25, 40
    pts_in = rng.uniform(-1, 1, (n_in * 8, 2))
    pts_in = pts_in[(pts_in ** 2).sum(axis=1) < 0.22][:n_in]
    pts_out = rng.uniform(-1, 1, (n_out * 4, 2))
    pts_out = pts_out[(pts_out ** 2).sum(axis=1) > 0.32][:n_out]

    cmap = LinearSegmentedColormap.from_list("region", ["#EFF6FF", "#BFDBFE", PRIMARY])

    fig, ax = new_figure(figsize=(6.4, 5.2))
    ax.contourf(x1, x2, prob, levels=np.linspace(0, 1, 21), cmap=cmap, alpha=0.75, zorder=1)
    cs = ax.contour(x1, x2, prob, levels=[0.5], colors=[INK], linewidths=1.6,
                    linestyles="--", zorder=3)
    ax.clabel(cs, fmt="输出 = 0.5", fontsize=10)
    ax.scatter(pts_out[:, 0], pts_out[:, 1], s=50, facecolor="white",
               edgecolor=PRIMARY, linewidth=1.6, zorder=4, label="圆外（标签 0）")
    ax.scatter(pts_in[:, 0], pts_in[:, 1], s=50, color=ACCENT, zorder=4,
               label="圆内（标签 1）")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    save_figure(fig, "ch17-circle-decision-boundary.svg")


if __name__ == "__main__":
    setup_style()
    plot_circle_boundary()
