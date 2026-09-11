# -*- coding: utf-8 -*-
"""生成第 5 章插图：下山示意、损失碗、抛物线采样点、梯度下降步骤、三种学习率。"""

import numpy as np

from fig_common import ACCENT, FILL, INK, MUTED, PRIMARY, new_figure, save_figure, setup_style


def _terrain(x):
    return 3.6 * np.exp(-((x - 2.6) ** 2) / 3.0) + 0.05 * (x - 8.2) ** 2


def _descent_history(w0, eta, steps):
    ws = [w0]
    for _ in range(steps):
        ws.append(ws[-1] - eta * 2 * ws[-1])
    return ws


def _step_arrows(ax, ws, rad=0.0):
    for prev, nxt in zip(ws, ws[1:]):
        ax.annotate("", xy=(nxt, nxt ** 2), xytext=(prev, prev ** 2),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, linewidth=1.6,
                                    shrinkA=4, shrinkB=4, mutation_scale=13,
                                    connectionstyle=f"arc3,rad={rad}"))


def plot_mountain():
    x = np.linspace(0, 10, 500)

    fig, ax = new_figure()
    ax.plot(x, _terrain(x), color=PRIMARY, linewidth=2.5, zorder=3)
    ax.fill_between(x, _terrain(x), color=FILL, alpha=0.7, zorder=1)
    px = 4.6
    py = _terrain(px)
    ax.scatter([px], [py], s=70, color=ACCENT, zorder=4)
    ax.annotate("你在这里", xy=(px, py), xytext=(px - 0.3, py + 1.1),
                fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    nx = 6.4
    ax.annotate("", xy=(nx, _terrain(nx) + 0.12), xytext=(px + 0.3, py + 0.3),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, linewidth=1.8,
                                connectionstyle="arc3,rad=-0.25", mutation_scale=16))
    ax.text(5.3, 2.7, "朝最陡的下坡方向迈一小步", fontsize=11, color=INK)
    vx = 8.2
    ax.scatter([vx], [_terrain(vx)], s=70, color=PRIMARY, zorder=4)
    ax.annotate("谷底：损失最小", xy=(vx, _terrain(vx)), xytext=(vx - 2.2, 1.2),
                fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlim(-0.3, 10.4)
    ax.set_ylim(-0.5, 6.0)
    ax.axis("off")
    save_figure(fig, "ch05-mountain-descent.svg")


def plot_bowl():
    w = np.linspace(-3, 3, 400)

    fig, ax = new_figure()
    ax.plot(w, w ** 2, color=PRIMARY, linewidth=2.5, zorder=3,
            label="$L(w, b) = w^2 + b^2$ 的横截面")
    ax.scatter([0], [0], s=55, color=ACCENT, zorder=4)
    ax.annotate("碗底：$w = 0,\\ b = 0$，损失最小", xy=(0, 0), xytext=(0.4, 1.8),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("w（固定 $b = 0$ 时的横截面）")
    ax.set_ylabel("L")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-0.4, 9.4)
    ax.legend(loc="upper center", frameon=False)
    save_figure(fig, "ch05-loss-bowl.svg")


def plot_loss_parabola():
    w = np.linspace(-0.15, 3.15, 400)

    fig, ax = new_figure()
    ax.plot(w, (2 * w - 3) ** 2, color=PRIMARY, linewidth=2.5, zorder=3,
            label="$L(w) = (2w - 3)^2$")
    sample_w = np.arange(0, 3.01, 0.5)
    sample_l = (2 * sample_w - 3) ** 2
    ax.scatter(sample_w, sample_l, s=45, color=ACCENT, zorder=4, label="正文表格采样点")
    for sw, sl in zip(sample_w, sample_l):
        ax.text(sw, sl + 0.35, f"{sl:g}", ha="center", fontsize=10, color=INK)
    ax.annotate("最低点 $w = 1.5$", xy=(1.5, 0), xytext=(2.1, 1.8),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("w")
    ax.set_ylabel("L")
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.5, 10)
    ax.legend(loc="upper center", frameon=False)
    save_figure(fig, "ch05-loss-parabola.svg")


def plot_descent_steps():
    w = np.linspace(0, 4.7, 400)
    steps = [4.0, 3.2, 2.56, 2.048]

    fig, ax = new_figure()
    ax.plot(w, w ** 2, color=PRIMARY, linewidth=2.5, zorder=3, label="$L(w) = w^2$")
    for i, sw in enumerate(steps):
        ax.scatter([sw], [sw ** 2], s=55, color=ACCENT, zorder=4)
        ax.text(sw + 0.09, sw ** 2 + 0.4, f"$w_{i}$", fontsize=11, color=INK)
    _step_arrows(ax, steps)
    wt = np.linspace(2.6, 4.2, 50)
    ax.plot(wt, 16 + 8 * (wt - 4), color=ACCENT, linewidth=1.4, linestyle="--", zorder=2)
    ax.annotate("起点处切线斜率 = 8（梯度）", xy=(3.0, 8.0), xytext=(0.4, 11.5),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(0.9, 2.6, "每一步落点都更低", fontsize=11, color=INK)
    ax.set_xlabel("w")
    ax.set_ylabel("L")
    ax.set_xlim(0, 4.9)
    ax.set_ylim(0, 17.8)
    ax.set_yticks([0, 4, 9, 16])
    ax.legend(loc="upper center", frameon=False)
    save_figure(fig, "ch05-descent-steps.svg")


def plot_lr_good():
    ws = _descent_history(1.0, 0.1, 5)
    w = np.linspace(0, 1.3, 400)

    fig, ax = new_figure()
    ax.plot(w, w ** 2, color=PRIMARY, linewidth=2.5, zorder=3, label="$L(w) = w^2$")
    for i, sw in enumerate(ws):
        ax.scatter([sw], [sw ** 2], s=55, color=ACCENT, zorder=4)
        ax.text(sw + 0.02, sw ** 2 + 0.04, f"$w_{i}$", fontsize=10, color=INK)
    _step_arrows(ax, ws)
    ax.text(0.42, 0.85, "$\\eta = 0.1$：稳步下降，步步接近谷底", fontsize=12, color=INK)
    ax.set_xlabel("w")
    ax.set_ylabel("L")
    ax.set_xlim(0, 1.32)
    ax.set_ylim(0, 1.3)
    save_figure(fig, "ch05-lr-good.svg")


def plot_lr_small():
    ws = _descent_history(1.0, 0.01, 5)
    w = np.linspace(0, 1.3, 400)

    fig, ax = new_figure()
    ax.plot(w, w ** 2, color=PRIMARY, linewidth=2.5, zorder=3, label="$L(w) = w^2$")
    for sw in ws:
        ax.scatter([sw], [sw ** 2], s=55, color=ACCENT, zorder=4)
    _step_arrows(ax, ws)
    ax.annotate("走了 5 步才挪这么点", xy=(0.93, 0.93 ** 2), xytext=(0.25, 0.85),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(0.3, 0.3, "$\\eta = 0.01$：方向对，但步子碎\n还要很久才能到谷底", fontsize=12, color=INK)
    ax.set_xlabel("w")
    ax.set_ylabel("L")
    ax.set_xlim(0, 1.32)
    ax.set_ylim(0, 1.3)
    save_figure(fig, "ch05-lr-small.svg")


def plot_lr_large():
    ws = _descent_history(1.0, 1.1, 3)
    w = np.linspace(-2.3, 2.3, 400)

    fig, ax = new_figure()
    ax.plot(w, w ** 2, color=PRIMARY, linewidth=2.5, zorder=3, label="$L(w) = w^2$")
    for i, sw in enumerate(ws):
        ax.scatter([sw], [sw ** 2], s=55, color=ACCENT, zorder=4)
    _step_arrows(ax, ws)
    ax.text(1.08, 0.78, "$w_0$", fontsize=11, color=INK)
    ax.text(-1.55, 1.5, "$w_1$", fontsize=11, color=INK)
    ax.text(1.5, 2.1, "$w_2$", fontsize=11, color=INK)
    ax.text(-2.05, 3.05, "$w_3$", fontsize=11, color=INK)
    ax.annotate("一步跨过谷底，弹到更高处", xy=(-1.2, 1.44), xytext=(-0.9, 2.6),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(-0.55, 0.35, "$\\eta = 1.1$：来回震荡，越走越远", fontsize=12, color=INK)
    ax.set_xlabel("w")
    ax.set_ylabel("L")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(0, 3.7)
    save_figure(fig, "ch05-lr-large.svg")


if __name__ == "__main__":
    setup_style()
    plot_mountain()
    plot_bowl()
    plot_loss_parabola()
    plot_descent_steps()
    plot_lr_good()
    plot_lr_small()
    plot_lr_large()
