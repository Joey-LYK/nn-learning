# -*- coding: utf-8 -*-
"""生成第 4 章插图：指数曲线、e^x 与 ln(x) 的对称、交叉熵损失、sigmoid。"""

import numpy as np

from fig_common import ACCENT, INK, MUTED, PRIMARY, new_figure, save_figure, setup_style


def plot_exp_2x():
    x = np.linspace(-1.5, 3.3, 400)

    fig, ax = new_figure()
    ax.plot(x, 2.0 ** x, color=PRIMARY, linewidth=2.5, zorder=3, label="$y = 2^x$")
    sample_x = np.array([-1, 0, 1, 2, 3], dtype=float)
    sample_y = 2.0 ** sample_x
    ax.scatter(sample_x, sample_y, s=45, color=ACCENT, zorder=4, label="采样点")
    for sx, sy in zip(sample_x, sample_y):
        ax.text(sx - 0.1, sy + 0.4, f"{sy:g}", fontsize=10, color=INK)
    ax.set_xlabel("x")
    ax.set_ylabel("$2^x$")
    ax.set_xlim(-1.6, 3.4)
    ax.set_ylim(0, 9.5)
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, "ch04-exp-2x.svg")


def plot_exp_log_symmetry():
    fig, ax = new_figure()
    x_exp = np.linspace(-2.2, 1.7, 400)
    ax.plot(x_exp, np.exp(x_exp), color=PRIMARY, linewidth=2.5, zorder=3, label="$y = e^x$")
    x_ln = np.linspace(0.03, 4.2, 400)
    ax.plot(x_ln, np.log(x_ln), color=ACCENT, linewidth=2.5, zorder=3, label="$y = \\ln(x)$")
    d = np.linspace(-2.2, 4.2, 100)
    ax.plot(d, d, color=MUTED, linewidth=1.2, linestyle="--", zorder=2, label="$y = x$")
    ax.scatter([0, 1], [1, 0], s=55, color=INK, zorder=4)
    ax.annotate("$e^x$ 穿过 $(0, 1)$", xy=(0, 1), xytext=(0.4, 2.6),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("$\\ln(x)$ 穿过 $(1, 0)$", xy=(1, 0), xytext=(2.3, -1.6),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-2.4, 4.4)
    ax.set_ylim(-2.4, 4.4)
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, "ch04-exp-log-symmetry.svg")


def plot_cross_entropy():
    p = np.linspace(0.02, 1.0, 400)

    fig, ax = new_figure()
    ax.plot(p, -np.log(p), color=PRIMARY, linewidth=2.5, zorder=3, label="$L = -\\ln(p)$")
    ax.scatter([1], [0], s=55, color=ACCENT, zorder=4)
    ax.annotate("预测自信且正确：损失为 0", xy=(1, 0), xytext=(0.45, 0.7),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("自信地犯错：损失暴涨", xy=(0.05, -np.log(0.05)), xytext=(0.2, 3.4),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("p（正确类别的预测概率）")
    ax.set_ylabel("损失 $-\\ln(p)$")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 4.2)
    ax.legend(loc="upper center", frameon=False)
    save_figure(fig, "ch04-cross-entropy-loss.svg")


def plot_sigmoid():
    x = np.linspace(-4.5, 4.5, 400)
    a = 1.0 / (1.0 + np.exp(-x))

    fig, ax = new_figure()
    ax.plot(x, a, color=PRIMARY, linewidth=2.5, zorder=3,
            label="$f(x) = \\dfrac{1}{1 + e^{-x}}$")
    ax.scatter([0], [0.5], s=55, color=ACCENT, zorder=4)
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
    ax.annotate("$f(0) = 0.5$", xy=(0, 0.5), xytext=(1.0, 0.28),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, "ch04-sigmoid.svg")


if __name__ == "__main__":
    setup_style()
    plot_exp_2x()
    plot_exp_log_symmetry()
    plot_cross_entropy()
    plot_sigmoid()
