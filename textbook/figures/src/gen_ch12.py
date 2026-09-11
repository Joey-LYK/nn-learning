# -*- coding: utf-8 -*-
"""生成第 12 章插图：sigmoid/tanh/ReLU 曲线、sigmoid 导数、三导数对比。"""

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


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def plot_sigmoid():
    z = np.linspace(-6, 6, 400)
    a = _sigmoid(z)

    fig, ax = new_figure()
    ax.plot(z, a, color=PRIMARY, linewidth=2.5, zorder=3)
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
    ax.scatter([0], [0.5], s=55, color=ACCENT, zorder=4)
    ax.annotate("$\\sigma(0) = 0.5$", xy=(0, 0.5), xytext=(1.4, 0.32),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(-5.6, 0.92, "两端变平：饱和区", fontsize=10, color=MUTED)
    ax.set_xlabel("z")
    ax.set_ylabel("$\\sigma(z)$")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    save_figure(fig, "ch12-sigmoid.svg")


def plot_tanh():
    z = np.linspace(-6, 6, 400)
    a = np.tanh(z)

    fig, ax = new_figure()
    ax.plot(z, a, color=PRIMARY, linewidth=2.5, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
    ax.scatter([0], [0], s=55, color=ACCENT, zorder=4)
    ax.annotate("过原点，输出以 0 为中心", xy=(0, 0), xytext=(0.9, -0.55),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("z")
    ax.set_ylabel("$\\tanh(z)$")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    save_figure(fig, "ch12-tanh.svg")


def plot_relu():
    z = np.linspace(-3, 3, 400)
    a = np.maximum(0, z)

    fig, ax = new_figure()
    ax.plot(z, a, color=PRIMARY, linewidth=2.5, zorder=3)
    ax.scatter([0], [0], s=55, color=ACCENT, zorder=4)
    ax.annotate("负数砍成 0，正数原样通过", xy=(0, 0), xytext=(-2.8, 1.3),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("z")
    ax.set_ylabel("ReLU($z$)")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-0.3, 3.2)
    save_figure(fig, "ch12-relu.svg")


def plot_sigmoid_derivative():
    z = np.linspace(-6, 6, 400)
    s = _sigmoid(z)
    ds = s * (1.0 - s)

    fig, ax = new_figure()
    ax.plot(z, ds, color=PRIMARY, linewidth=2.5, zorder=3)
    ax.scatter([0], [0.25], s=55, color=ACCENT, zorder=4)
    ax.annotate("峰值只有 0.25（$z = 0$ 处）", xy=(0, 0.25), xytext=(1.2, 0.21),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(-5.9, 0.018, "两端贴地：$|z| > 5$ 梯度几乎断流", fontsize=10, color=MUTED)
    ax.set_xlabel("z")
    ax.set_ylabel("$\\sigma'(z) = \\sigma(z)\\,(1-\\sigma(z))$")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(0, 0.29)
    save_figure(fig, "ch12-sigmoid-derivative.svg")


def plot_derivative_compare():
    z = np.linspace(-6, 6, 600)
    s = _sigmoid(z)
    ds = s * (1.0 - s)
    dt = 1.0 - np.tanh(z) ** 2

    fig, ax = new_figure()
    ax.plot(z, dt, color=ACCENT, linewidth=2.2, zorder=3,
            label="Tanh 导数：峰值 1.0")
    ax.plot(z, ds, color=PRIMARY, linewidth=2.2, zorder=4,
            label="Sigmoid 导数：峰值只有 0.25")
    ax.plot([-6, 0], [0, 0], color=INK, linewidth=2.5, zorder=5)
    ax.plot([0, 6], [1, 1], color=INK, linewidth=2.5, zorder=5,
            label="ReLU 导数：$z > 0$ 恒为 1")
    ax.scatter([0], [1], s=45, color=INK, zorder=6)
    ax.scatter([0], [0.25], s=45, color=PRIMARY, zorder=6)
    ax.set_xlabel("z")
    ax.set_ylabel("导数")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 1.0])
    ax.legend(loc="center left", frameon=False, fontsize=10)
    save_figure(fig, "ch12-derivative-compare.svg")


if __name__ == "__main__":
    setup_style()
    plot_sigmoid()
    plot_tanh()
    plot_relu()
    plot_sigmoid_derivative()
    plot_derivative_compare()
