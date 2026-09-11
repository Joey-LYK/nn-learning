# -*- coding: utf-8 -*-
"""生成第 19 章插图：梯度沿时间回传按 (1/4)^n 衰减——长期依赖问题。"""

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


def plot_gradient_decay():
    fig, ax = new_figure()
    n = np.arange(1, 21)
    decay = 0.25 ** n

    ax.semilogy(n, np.ones_like(n, dtype=float), color=MUTED, linewidth=1.6,
                linestyle="--", label="理想情况：梯度不衰减")
    ax.semilogy(n, decay, color=PRIMARY, linewidth=2.4, marker="o", markersize=4,
                label="现实：每回传一步，梯度 ×1/4")

    for step, dy in ((5, 1.6), (10, 1.6), (20, 0.35)):
        v = 0.25 ** step
        ax.annotate(f"{v:.0e}".replace("e-0", "e−"), xy=(step, v),
                    xytext=(step + 0.3, v * dy), fontsize=9, color=ACCENT)
    ax.annotate("传到第 1 层时已只剩万亿分之一，\n权重几乎收不到更新信号",
                xy=(20, 0.25 ** 20), xytext=(11.2, 1e-4),
                fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.0))

    ax.set_xlabel("梯度要回传的步数 n")
    ax.set_ylabel("传到第 1 层的梯度（对数刻度）")
    ax.set_xlim(0.5, 21.5)
    ax.set_ylim(1e-13, 5)
    ax.set_xticks(np.arange(1, 21, 2))
    ax.legend(loc="upper right", fontsize=9)
    save_figure(fig, "ch19-gradient-decay.svg")


if __name__ == "__main__":
    setup_style()
    plot_gradient_decay()
