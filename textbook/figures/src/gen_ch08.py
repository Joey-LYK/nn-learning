# -*- coding: utf-8 -*-
"""生成第 8 章插图：交叉熵曲线、错题数阶梯图、损失碗。"""

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


def plot_cross_entropy():
    p = np.linspace(0.01, 1.0, 400)
    loss = -np.log(p)

    fig, ax = new_figure()
    ax.plot(p, loss, color=PRIMARY, linewidth=2.5, zorder=3)
    sample_p = [0.01, 0.1, 0.55, 0.9]
    sample_l = -np.log(np.array(sample_p))
    ax.scatter(sample_p, sample_l, s=45, color=ACCENT, zorder=4, label="正文表格取值")
    for pv, lv, name in zip(sample_p, sample_l, ["4.605", "2.303", "0.598", "0.105"]):
        ax.annotate(f"{name}", xy=(pv, lv), xytext=(pv + 0.04, lv + 0.22),
                    fontsize=10, color=INK)
    ax.set_xlabel("p（你给正确答案的概率）")
    ax.set_ylabel("$L = -\\ln p$")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.15, 5.0)
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "ch08-cross-entropy.svg")


def plot_error_steps():
    fig, ax = new_figure()
    ax.step([0, 3.3, 6.7, 10], [2, 3, 4, 4], where="post",
            color=PRIMARY, linewidth=2.5, zorder=3)
    ax.scatter([3.3, 6.7], [2, 3], s=45, facecolor="white",
               edgecolor=PRIMARY, linewidth=1.6, zorder=4)
    ax.annotate("平坦段：导数 = 0，给不出调参方向", xy=(5.0, 3), xytext=(2.6, 4.35),
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("跳变点：导数不存在", xy=(3.3, 2.5), xytext=(3.8, 1.2),
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("某个权重 $w$")
    ax.set_ylabel("错题数")
    ax.set_xlim(-0.3, 10.4)
    ax.set_ylim(0, 5.2)
    ax.set_yticks([0, 1, 2, 3, 4])
    save_figure(fig, "ch08-error-steps.svg")


def plot_loss_bowl():
    w = np.linspace(-2.2, 2.2, 400)
    loss = w ** 2

    fig, ax = new_figure()
    ax.plot(w, loss, color=PRIMARY, linewidth=2.5, zorder=3)
    ax.scatter([0], [0], s=55, color=ACCENT, zorder=4)
    ax.annotate("谷底：$L$ 最小，权重最优", xy=(0, 0), xytext=(0.25, 1.6),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("权重 $w$")
    ax.set_ylabel("损失 $L$")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.35, 5.2)
    ax.set_xticks([])
    ax.set_yticks([])
    save_figure(fig, "ch08-loss-bowl.svg")


if __name__ == "__main__":
    setup_style()
    plot_cross_entropy()
    plot_error_steps()
    plot_loss_bowl()
