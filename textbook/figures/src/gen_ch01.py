# -*- coding: utf-8 -*-
"""生成第 1 章插图：函数机器流程、三个散点图、割线与切线。"""

import numpy as np
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


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=16))


def plot_function_machine():
    fig, ax = new_figure(figsize=(7.2, 2.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [(2.0, "输入 $x$"), (6.0, "函数机器 $f$"), (10.0, "输出 $f(x)$")]
    for cx, text in boxes:
        ax.add_patch(FancyBboxPatch((cx - 1.35, 1.35), 2.7, 1.3,
                                    boxstyle="round,pad=0.15",
                                    facecolor=FILL, edgecolor=PRIMARY, linewidth=1.8))
        ax.text(cx, 2.0, text, ha="center", va="center", fontsize=13, color=INK)
    _arrow(ax, (3.55, 2.0), (4.45, 2.0))
    _arrow(ax, (7.55, 2.0), (8.45, 2.0))

    save_figure(fig, "ch01-function-machine.svg")


def plot_linear_points():
    x = np.array([-2, -1, 0, 1, 2])
    y = 2 * x + 1

    fig, ax = new_figure()
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.plot(x, y, color=PRIMARY, linewidth=1.5, linestyle="--", zorder=3)
    ax.scatter(x, y, s=45, color=ACCENT, zorder=4)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-4.2, 6.2)
    ax.set_xticks(x)
    ax.set_yticks(y)
    save_figure(fig, "ch01-linear-points.svg")


def plot_square_points():
    x = np.array([-2, -1, 0, 1, 2, 3])
    y = x ** 2

    fig, ax = new_figure()
    ax.scatter(x, y, s=45, color=ACCENT, zorder=4)
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.set_xlim(-2.6, 3.6)
    ax.set_ylim(-0.8, 10.2)
    ax.set_xticks(x)
    ax.set_yticks([0, 1, 4, 9])
    save_figure(fig, "ch01-square-points.svg")


def plot_constant_points():
    x = np.arange(5)
    y = np.full(5, 5)

    fig, ax = new_figure()
    ax.axhline(5, color=MUTED, linewidth=1.5, linestyle="--", zorder=2)
    ax.scatter(x, y, s=45, color=ACCENT, zorder=4)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(0, 6.4)
    ax.set_xticks(x)
    ax.set_yticks([5])
    save_figure(fig, "ch01-constant.svg")


def plot_secant():
    x = np.linspace(-1.2, 3.6, 400)
    y = x ** 2

    fig, ax = new_figure()
    ax.plot(x, y, color=PRIMARY, linewidth=2.5, zorder=3)

    xs = np.array([0.9, 3.5])
    ax.plot(xs, 5 * xs - 6, color=ACCENT, linewidth=2.0, zorder=3)
    xt = np.array([1.0, 3.4])
    ax.plot(xt, 4 * xt - 4, color=MUTED, linewidth=1.8, linestyle="--", zorder=3)

    ax.scatter([2, 3], [4, 9], s=50, color=ACCENT, zorder=4)
    ax.annotate("割线：连接 $(2,4)$ 和远处一点", xy=(3.35, 10.75), xytext=(0.6, 11.6),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("切线：贴着 $(2,4)$ 恰好擦过", xy=(1.9, 3.6), xytext=(-1.0, 6.8),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("$(2,4)$", xy=(2, 4), xytext=(2.25, 2.6), fontsize=11, color=INK)
    ax.annotate("$(3,9)$", xy=(3, 9), xytext=(2.55, 9.6), fontsize=11, color=INK)

    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.set_xlim(-1.5, 4.0)
    ax.set_ylim(-2.5, 13.5)
    save_figure(fig, "ch01-secant.svg")


if __name__ == "__main__":
    setup_style()
    plot_function_machine()
    plot_linear_points()
    plot_square_points()
    plot_constant_points()
    plot_secant()
