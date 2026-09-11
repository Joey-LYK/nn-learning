# -*- coding: utf-8 -*-
"""生成第 10 章插图：前向存值 / 反向摊责对照流程图。"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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


def _box(ax, cx, cy, w, h, text, edge=PRIMARY, fontsize=11):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.06",
                                facecolor=FILL, edgecolor=edge,
                                linewidth=1.8, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=4)


def _flow_arrow(ax, start, end, color=INK):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=16))


def plot_forward_backward():
    fig, ax = new_figure(figsize=(9.6, 4.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")

    y_fwd, y_bwd = 7.4, 3.0
    xs = [2.2, 4.6, 7.0, 9.4, 11.6, 13.6]
    labels = ["$x$", "$z_1, z_2$", "$a_1, a_2$", "$z_3$", "$\\hat{y}$", "$L$"]
    box_w, box_h = 1.7, 1.2

    ax.text(0.35, y_fwd + 1.5, "前向(→)", ha="left", va="center", fontsize=12, color=PRIMARY)
    for i, (cx, text) in enumerate(zip(xs, labels)):
        _box(ax, cx, y_fwd, box_w, box_h, text)
        if i > 0:
            _flow_arrow(ax, (xs[i - 1] + box_w / 2, y_fwd), (cx - box_w / 2, y_fwd))
    for cx in xs[1:5]:
        ax.text(cx, y_fwd - box_h / 2 - 0.55, "存", ha="center", va="center",
                fontsize=11, color=MUTED)

    ax.text(0.35, y_bwd + 1.5, "反向(←)", ha="left", va="center", fontsize=12, color=ACCENT)
    bwd = [(11.6, "$\\hat{y} - y$", 1.9), (9.4, "$\\delta_o$", 1.9),
           (4.6, "$\\delta_{h_1}, \\delta_{h_2}$", 3.2)]
    for cx, text, w in bwd:
        _box(ax, cx, y_bwd, w, box_h, text, edge=ACCENT)
    _flow_arrow(ax, (11.6 - 1.9 / 2, y_bwd), (9.4 + 1.9 / 2, y_bwd), ACCENT)
    _flow_arrow(ax, (9.4 - 1.9 / 2, y_bwd), (4.6 + 3.2 / 2, y_bwd), ACCENT)
    ax.text(10.5, y_bwd + 0.95, "$\\partial L/\\partial \\hat{y}$",
            ha="center", fontsize=10, color=MUTED)

    _flow_arrow(ax, (9.4, y_bwd - box_h / 2), (9.4, 1.35), ACCENT)
    ax.text(9.4, 0.85, "$\\partial L/\\partial v_1, \\partial L/\\partial v_2 = \\delta_o \\times a$",
            ha="center", fontsize=10, color=INK)
    _flow_arrow(ax, (4.6, y_bwd - box_h / 2), (4.6, 1.35), ACCENT)
    ax.text(4.6, 0.85, "$\\partial L/\\partial w_{11} \\cdots w_{22} = \\delta_h \\times x$",
            ha="center", fontsize=10, color=INK)

    save_figure(fig, "ch10-forward-backward.svg")


if __name__ == "__main__":
    setup_style()
    plot_forward_backward()
