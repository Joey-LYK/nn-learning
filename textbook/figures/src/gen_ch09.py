# -*- coding: utf-8 -*-
"""生成第 9 章插图：sin 拟合 loss 曲线、XOR 样本分布。"""

from fig_common import (
    ACCENT,
    INK,
    MUTED,
    PRIMARY,
    new_figure,
    save_figure,
    setup_style,
)


def plot_loss_curve():
    epochs = [1, 800, 1600, 2400, 3200, 4000]
    losses = [0.293546, 0.024993, 0.007556, 0.003157, 0.005512, 0.001037]

    fig, ax = new_figure()
    ax.plot(epochs, losses, color=PRIMARY, linewidth=2.0, zorder=3)
    ax.scatter(epochs, losses, s=45, color=ACCENT, zorder=4, label="正文打印的 loss")
    ax.annotate("前 800 轮降掉约 90%", xy=(800, 0.024993), xytext=(1250, 0.14),
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("小幅抖动属正常（抽样噪声）", xy=(3200, 0.005512), xytext=(2500, 0.06),
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("epoch（训练轮次）")
    ax.set_ylabel("loss（MSE）")
    ax.set_xlim(-150, 4200)
    ax.set_ylim(-0.012, 0.32)
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "ch09-loss-curve.svg")


def plot_xor_samples():
    zeros = [(0.0, 0.0), (1.0, 1.0)]
    ones = [(0.0, 1.0), (1.0, 0.0)]

    fig, ax = new_figure(figsize=(5.2, 4.2))
    ax.scatter(*zip(*zeros), s=130, marker="o", color=PRIMARY,
               edgecolor=INK, linewidth=1.2, zorder=4, label="输出 0")
    ax.scatter(*zip(*ones), s=130, marker="^", color=ACCENT,
               edgecolor=INK, linewidth=1.2, zorder=4, label="输出 1")
    ax.text(0.5, -0.16, "一条直线无法把两类分开（线性不可分）",
            ha="center", fontsize=10, color=MUTED)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "ch09-xor-samples.svg")


if __name__ == "__main__":
    setup_style()
    plot_loss_curve()
    plot_xor_samples()
