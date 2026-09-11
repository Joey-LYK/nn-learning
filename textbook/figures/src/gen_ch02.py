# -*- coding: utf-8 -*-
"""生成第 2 章插图：平面上的向量箭头。"""

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


def plot_vector_arrow():
    fig, ax = new_figure(figsize=(5.2, 4.0))
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.annotate("", xy=(3, 4), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PRIMARY, linewidth=2.5,
                                shrinkA=0, shrinkB=0, mutation_scale=20), zorder=3)
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


if __name__ == "__main__":
    setup_style()
    plot_vector_arrow()
