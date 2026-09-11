# -*- coding: utf-8 -*-
"""生成第 2 章插图：平面上的向量箭头、向量加法与数乘。"""

import matplotlib.pyplot as plt
import numpy as np

from fig_common import (
    ACCENT,
    INK,
    MUTED,
    PRIMARY,
    new_figure,
    save_figure,
    setup_style,
)


def _axes_frame(ax):
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.set_aspect("equal")


def _arrow(ax, xy, xytext, color, linewidth=2.5, dashed=False):
    ax.annotate("", xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=linewidth,
                                shrinkA=0, shrinkB=0, mutation_scale=18,
                                linestyle=(0, (4, 3)) if dashed else "solid"),
                zorder=3)


def plot_vector_arrow():
    fig, ax = new_figure(figsize=(5.2, 4.0))
    _axes_frame(ax)
    _arrow(ax, (3, 4), (0, 0), PRIMARY)
    ax.scatter([0], [0], s=40, color=INK, zorder=4)
    ax.scatter([3], [4], s=55, color=ACCENT, zorder=4)
    ax.text(3.18, 4.0, "$(3,4)$", fontsize=12, color=INK, va="center")
    ax.text(0.85, 2.35, "向量 $[3,4]$", fontsize=12, color=PRIMARY)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(-0.4, 5.0)
    ax.set_xticks(np.arange(0, 4))
    ax.set_yticks(np.arange(0, 5, 2))
    save_figure(fig, "ch02-vector-arrow.svg")


def plot_vector_addition():
    """左：[2,3] + [1,4] 首尾相接得 [3,7]；右：数乘 3 把箭头拉长到 [6,9]。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.6))
    _axes_frame(ax1)
    _axes_frame(ax2)

    # 左图：加法
    _arrow(ax1, (2, 3), (0, 0), PRIMARY)
    _arrow(ax1, (3, 7), (2, 3), ACCENT)
    _arrow(ax1, (3, 7), (0, 0), INK, linewidth=1.8, dashed=True)
    ax1.scatter([0, 2, 3], [0, 3, 7], s=32, color=INK, zorder=4)
    ax1.text(0.85, 1.02, "$[2,3]$", fontsize=11, color=PRIMARY)
    ax1.text(2.6, 5.55, "$[1,4]$", fontsize=11, color=ACCENT)
    ax1.text(3.25, 7.15, "$[3,7]$", fontsize=11, color=INK)
    ax1.set_title("加法：首尾相接", fontsize=11, color=INK)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(-0.5, 4.3)
    ax1.set_ylim(-0.6, 8.2)
    ax1.set_xticks(np.arange(0, 5))
    ax1.set_yticks(np.arange(0, 9, 2))

    # 右图：数乘
    _arrow(ax2, (2, 3), (0, 0), PRIMARY)
    _arrow(ax2, (6, 9), (0, 0), ACCENT, linewidth=2.8)
    ax2.plot([0, 6.9], [0, 10.35], color=MUTED, linewidth=1.0,
             linestyle=":", zorder=2)
    ax2.scatter([0, 2, 6], [0, 3, 9], s=32, color=INK, zorder=4)
    ax2.text(0.55, 1.75, "$[2,3]$", fontsize=11, color=PRIMARY)
    ax2.text(3.9, 9.55, "$3\\times[2,3]=[6,9]$", fontsize=11, color=ACCENT)
    ax2.text(5.6, 5.4, "同一方向，拉长 3 倍", fontsize=9.5, color=MUTED,
             rotation=0, ha="right")
    ax2.set_title("数乘：拉长 3 倍", fontsize=11, color=INK)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(-0.5, 7.6)
    ax2.set_ylim(-0.6, 10.6)
    ax2.set_xticks(np.arange(0, 8, 2))
    ax2.set_yticks(np.arange(0, 11, 2))
    save_figure(fig, "ch02-vector-addition.svg")


if __name__ == "__main__":
    setup_style()
    plot_vector_arrow()
    plot_vector_addition()
