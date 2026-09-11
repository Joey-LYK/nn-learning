# -*- coding: utf-8 -*-
"""生成第 6 章插图：生物神经元、人工神经元、sigmoid 曲线、ReLU 曲线。"""

import matplotlib.pyplot as plt
import numpy as np
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


def plot_sigmoid():
    z = np.linspace(-6, 6, 400)
    a = 1.0 / (1.0 + np.exp(-z))

    fig, ax = new_figure()
    ax.plot(z, a, color=PRIMARY, linewidth=2.5, zorder=3)
    sample_z = [-4, -1, 0, 1, 4]
    sample_a = 1.0 / (1.0 + np.exp(-np.array(sample_z, dtype=float)))
    ax.scatter(sample_z, sample_a, s=45, color=ACCENT, zorder=4, label="表格采样点")
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
    ax.annotate("σ(0) = 0.5", xy=(0, 0.5), xytext=(1.2, 0.38),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("z（加权求和结果）")
    ax.set_ylabel("a = σ(z)")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "ch06-sigmoid.svg")


def plot_relu():
    z = np.linspace(-2.5, 2.5, 400)
    a = np.maximum(0, z)

    fig, ax = new_figure()
    ax.plot(z, a, color=PRIMARY, linewidth=2.5, zorder=3)
    ax.scatter([0], [0], s=55, color=ACCENT, zorder=4)
    ax.annotate("折点：负的归零，正的不动", xy=(0, 0), xytext=(-2.3, 1.1),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("z")
    ax.set_ylabel("a = ReLU(z) = max(0, z)")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-0.3, 2.6)
    save_figure(fig, "ch06-relu.svg")


def _circle_node(ax, xy, r, text, fontsize=13):
    ax.add_patch(Circle(xy, r, facecolor=FILL, edgecolor=PRIMARY, linewidth=1.8, zorder=3))
    ax.text(*xy, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=4)


def _arrow(ax, start, end, label=None, label_offset=(0, 0.45)):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=16))
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=12, color=ACCENT)


def plot_artificial_neuron():
    fig, ax = new_figure(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    inputs = [("x$_1$", 1.1, 8.0, r"$\times\,w_1$"),
              ("x$_2$", 1.1, 5.0, r"$\times\,w_2$"),
              ("x$_3$", 1.1, 2.0, r"$\times\,w_3$")]
    sum_xy = (4.6, 5.0)
    act_xy = (7.2, 5.0)

    for name, x, y, weight in inputs:
        _circle_node(ax, (x, y), 0.62, name)
        _arrow(ax, (x + 0.62, y), (sum_xy[0] - 0.75, sum_xy[1] + (y - sum_xy[1]) * 0.28), weight)
        ax.text(x - 0.95, y, "输入", ha="center", va="center", fontsize=10, color=MUTED)

    _circle_node(ax, sum_xy, 0.75, "Σ")
    _arrow(ax, (sum_xy[0] + 0.75, sum_xy[1]), (act_xy[0] - 0.75, act_xy[1]), "z")
    _circle_node(ax, act_xy, 0.75, "f")
    _arrow(ax, (act_xy[0] + 0.75, act_xy[1]), (act_xy[0] + 2.1, act_xy[1]), "a = f(z)", (0, 0.6))
    ax.text(act_xy[0] + 2.45, act_xy[1], "输出", ha="center", va="center", fontsize=11, color=MUTED)

    ax.text(sum_xy[0], sum_xy[1] - 2.55, "加权求和", ha="center", fontsize=10, color=MUTED)
    _arrow(ax, (sum_xy[0], 2.9), (sum_xy[0], sum_xy[1] - 0.75), "+b", (-0.5, 0))
    ax.text(act_xy[0], act_xy[1] - 1.6, "激活函数", ha="center", fontsize=10, color=MUTED)

    save_figure(fig, "ch06-artificial-neuron.svg")


def plot_biological_neuron():
    fig, ax = new_figure(figsize=(7.2, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    body = (4.0, 5.0)
    dendrite_tips = [(0.6, 8.6), (0.3, 5.0), (0.6, 1.4)]
    for tip in dendrite_tips:
        ax.annotate("", xy=(body[0] - 0.95, body[1] + (tip[1] - body[1]) * 0.25),
                    xytext=tip,
                    arrowprops=dict(arrowstyle="-|>", color=PRIMARY, linewidth=1.8,
                                    connectionstyle="arc3,rad=0.15", mutation_scale=16))
    ax.text(1.6, 8.9, "树突（接收信号）", fontsize=11, color=INK)

    ax.add_patch(Circle(body, 1.15, facecolor=FILL, edgecolor=PRIMARY, linewidth=1.8))
    ax.text(body[0], body[1] + 0.25, "细胞体", ha="center", va="center", fontsize=12, color=INK)
    ax.text(body[0], body[1] - 0.35, "（汇总）", ha="center", va="center", fontsize=10, color=INK)

    ax.plot([body[0] + 1.15, 9.0], [5.0, 5.0], color=PRIMARY, linewidth=2.2,
            solid_capstyle="round", zorder=2)
    ax.text(7.0, 5.55, "轴突（发送信号）", fontsize=11, color=INK)

    for dy in (0.55, 0, -0.55):
        ax.annotate("", xy=(11.2, 5.0 + dy * 1.6), xytext=(9.0, 5.0),
                    arrowprops=dict(arrowstyle="-|>", color=PRIMARY, linewidth=1.6, mutation_scale=14))
    ax.text(10.05, 7.6, "突触 → 传给\n下一个神经元", ha="center", fontsize=10, color=MUTED)

    save_figure(fig, "ch06-biological-neuron.svg")


if __name__ == "__main__":
    setup_style()
    plot_biological_neuron()
    plot_artificial_neuron()
    plot_sigmoid()
    plot_relu()
