# -*- coding: utf-8 -*-
"""生成第 7 章插图：2-2-1 网络前向传播数值流程图。"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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


def _circle_node(ax, xy, r, text, fontsize=13):
    ax.add_patch(Circle(xy, r, facecolor=FILL, edgecolor=PRIMARY, linewidth=1.8, zorder=3))
    ax.text(*xy, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=4)


def _arrow(ax, start, end, label=None, label_offset=(0, 0.45), t=0.5):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=16))
    if label:
        mx = start[0] + (end[0] - start[0]) * t + label_offset[0]
        my = start[1] + (end[1] - start[1]) * t + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=11, color=ACCENT)


def plot_forward_flow():
    fig, ax = new_figure(figsize=(8.4, 4.4))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 10)
    ax.axis("off")

    x1, x2 = (1.4, 7.4), (1.4, 2.6)
    h1, h2 = (6.3, 7.4), (6.3, 2.6)
    o = (11.3, 5.0)
    r_in, r_h, r_o = 0.72, 1.15, 1.25

    _circle_node(ax, x1, r_in, "$x_1$\n0.5", fontsize=11)
    _circle_node(ax, x2, r_in, "$x_2$\n0.8", fontsize=11)
    ax.text(x1[0], x1[1] - 1.45, "输入", ha="center", fontsize=10, color=MUTED)

    _circle_node(ax, h1, r_h, "$z_1$=1.02\n$a_1$=0.7350", fontsize=10)
    _circle_node(ax, h2, r_h, "$z_2$=0.13\n$a_2$=0.5325", fontsize=10)
    ax.text(h1[0], h1[1] + r_h + 0.5, "隐藏①  $b_1$=0.1", ha="center", fontsize=10, color=MUTED)
    ax.text(h2[0], h2[1] - r_h - 0.6, "隐藏②  $b_2$=-0.2", ha="center", fontsize=10, color=MUTED)

    _circle_node(ax, o, r_o, "$z_3$=0.4482\n$\\hat{y}$=0.6102", fontsize=10)
    ax.text(o[0], o[1] - r_o - 0.6, "输出  $b_3$=0.2", ha="center", fontsize=10, color=MUTED)

    _arrow(ax, (x1[0] + r_in, x1[1]), (h1[0] - r_h, h1[1] + 0.15),
           "$w_{11}$=0.4", (0.1, 0.4))
    _arrow(ax, (x2[0] + r_in, x2[1]), (h1[0] - r_h + 0.35, h1[1] - r_h * 0.75),
           "$w_{12}$=0.9", (-0.25, -0.35), t=0.32)
    _arrow(ax, (x1[0] + r_in, x1[1] - 0.35), (h2[0] - r_h + 0.35, h2[1] + r_h * 0.75),
           "$w_{21}$=-0.3", (-0.25, 0.35), t=0.32)
    _arrow(ax, (x2[0] + r_in, x2[1]), (h2[0] - r_h, h2[1] - 0.15),
           "$w_{22}$=0.6", (0.1, -0.4))
    _arrow(ax, (h1[0] + r_h, h1[1] - 0.35), (o[0] - r_o + 0.2, o[1] + r_o * 0.7),
           "$v_1$=0.7", (0, 0.45))
    _arrow(ax, (h2[0] + r_h, h2[1] + 0.35), (o[0] - r_o + 0.2, o[1] - r_o * 0.7),
           "$v_2$=-0.5", (0, -0.45))

    save_figure(fig, "ch07-forward-flow.svg")


if __name__ == "__main__":
    setup_style()
    plot_forward_flow()
