# -*- coding: utf-8 -*-
"""生成第 11 章插图：三种梯度下降损失曲线、动量法山谷俯视图、学习率阶梯衰减。"""

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


def plot_loss_curves():
    t = np.linspace(0, 100, 400)
    rng = np.random.default_rng(7)
    batch = 1.0 - 0.0045 * t
    mini = 0.85 * np.exp(-t / 35) + 0.12 + rng.normal(0, 0.012, t.size)
    sgd = 0.85 * np.exp(-t / 22) + 0.06 + rng.normal(0, 0.05, t.size)

    fig, ax = new_figure()
    ax.plot(t, batch, color=MUTED, linewidth=2.5, zorder=3, label="批量：稳但慢，每一步都很贵")
    ax.plot(t, mini, color=PRIMARY, linewidth=2.5, zorder=4, label="小批量：轻微抖动，稳定向下")
    ax.plot(t, sgd, color=ACCENT, linewidth=1.6, zorder=3, label="随机（单条）：抖动剧烈，大方向仍向下")
    ax.set_xlabel("时间")
    ax.set_ylabel("损失")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    save_figure(fig, "ch11-loss-curves.svg")


def plot_momentum_valley():
    x = np.linspace(-1.3, 1.3, 300)
    y = np.linspace(-1.15, 1.15, 300)
    X, Y = np.meshgrid(x, y)
    L = X ** 2 + 40.0 * Y ** 2

    fig, ax = new_figure(figsize=(6.4, 4.0))
    ax.contour(X, Y, L, levels=[0.2, 0.8, 2.0, 4.0, 8.0, 16.0, 32.0],
               colors=MUTED, linewidths=1.0)

    n = 16
    t = np.arange(n)
    gd_x = -1.0 + 0.062 * t
    gd_y = 0.95 * (-0.72) ** t
    mom_x = -1.0 + 0.070 * t
    mom_y = 0.95 * np.exp(-t / 3.2)

    ax.plot(gd_x, gd_y, color=PRIMARY, linewidth=2.2, marker="o", markersize=3.5,
            zorder=4, label="普通 GD：在两壁间震荡")
    ax.plot(mom_x, mom_y, color=ACCENT, linewidth=2.2, marker="o", markersize=3.5,
            zorder=5, label="动量法：震荡被抵消，直冲谷底")
    for xs, ys, color in ((gd_x, gd_y, PRIMARY), (mom_x, mom_y, ACCENT)):
        mid = n // 2
        ax.annotate("", xy=(xs[mid + 1], ys[mid + 1]), xytext=(xs[mid], ys[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.0,
                                    mutation_scale=18), zorder=7)
    ax.scatter([0], [0], s=110, color=ACCENT, marker="o", zorder=6,
               facecolors="none", linewidths=2.2)
    ax.text(0.06, -0.16, "谷底", fontsize=11, color=ACCENT)
    ax.set_xlabel("$x$ 方向（缓，通向谷底）")
    ax.set_ylabel("$y$ 方向（陡，来回震荡）")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    save_figure(fig, "ch11-momentum-valley.svg")


def plot_lr_step_decay():
    edges = [0, 1000, 2000, 3000, 4000]
    etas = [0.5, 0.25, 0.125, 0.0625]

    fig, ax = new_figure()
    ax.step(edges + [5000], etas + [etas[-1]] * 2, where="post",
            color=PRIMARY, linewidth=2.5, zorder=3)
    for left, eta in zip(edges, etas):
        ax.text(left + 500, eta + 0.018, f"$\\eta$ = {eta}",
                ha="center", fontsize=10, color=INK)
    ax.annotate("每过 1000 轮砍一半", xy=(3050, 0.0625), xytext=(1800, 0.36),
                fontsize=11, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("学习率 $\\eta$")
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 0.6)
    ax.set_xticks(edges + [5000])
    save_figure(fig, "ch11-lr-step-decay.svg")


if __name__ == "__main__":
    setup_style()
    plot_loss_curves()
    plot_momentum_valley()
    plot_lr_step_decay()
