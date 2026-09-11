# -*- coding: utf-8 -*-
"""生成第 21 章插图：迁移微调与从零训练在同一训练点之外的外推对比。"""

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


def plot_extrapolation():
    fig, ax = new_figure()
    x = np.linspace(0, 100, 200)

    ax.plot(x, 2 * x + 1, color=PRIMARY, linewidth=2.4,
            label="迁移微调：$y = 2x + 1$（$w$ 冻结）")
    ax.plot(x, 2.1 * x + 0.7, color=ACCENT, linewidth=2.4, linestyle="--",
            label="从零训练：$y \\approx 2.1x + 0.7$")

    ax.axhline(201, color=MUTED, linewidth=1.2, linestyle=":")
    ax.text(56, 205, "x = 100 处的真值 201", fontsize=9.5, color=INK)

    ax.scatter([3], [7], s=60, color=INK, zorder=5)
    ax.annotate("唯一的训练样本 (3, 7)\n两个方案在这里都零误差",
                xy=(3, 7), xytext=(30, 20), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.0))

    ax.scatter([100, 100], [201, 210.7], s=48,
               color=[PRIMARY, ACCENT], zorder=5)
    # 端点标签放到绘图区右缘之外，避免与两条汇聚的线相撞
    ax.annotate("迁移：201.0", xy=(100, 201), xytext=(1.03, 0.78),
                textcoords="axes fraction", fontsize=9.5, color=PRIMARY,
                va="center", annotation_clip=False)
    ax.annotate("从零：210.7\n（x 越大偏得越远）", xy=(100, 210.7),
                xytext=(1.03, 0.92), textcoords="axes fraction",
                fontsize=9.5, color=ACCENT, va="center", annotation_clip=False)

    ax.set_xlabel("x")
    ax.set_ylabel("预测值 y")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 232)
    ax.legend(loc="upper left", fontsize=9.5)
    save_figure(fig, "ch21-extrapolation.svg")


if __name__ == "__main__":
    setup_style()
    plot_extrapolation()
