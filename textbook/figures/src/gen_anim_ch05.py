# -*- coding: utf-8 -*-
"""生成第 5 章动画：小球沿 L = w² 的碗一步步梯度下降（η = 0.1，从 w = 4 出发）。"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from anim_common import save_gif
from fig_common import ACCENT, INK, MUTED, PRIMARY, setup_style

ETA = 0.1
STEPS = 14
HOLD = 4

# 预计算每一步的 w（与第 5 章实验一的确定性迭代完全一致）
ws = [4.0]
for _ in range(STEPS):
    w = ws[-1]
    ws.append(w - ETA * 2 * w)


def draw_frame(ax, f):
    step, phase = divmod(f, 2)          # 偶数帧=第 step 步起点，奇数帧=球滚向下一步
    step = min(step, STEPS)
    w_now = ws[step]
    w_next = ws[min(step + 1, STEPS)]
    t = phase / 2 + (0 if phase == 0 else 0.5) / 2
    w_ball = w_now + (w_next - w_now) * min(f * 0.5 - step, 1.0) if f > 0 else ws[0]

    ax.clear()
    grid = np.linspace(-4.6, 4.6, 200)
    ax.plot(grid, grid ** 2, color=MUTED, linewidth=2.0, zorder=2)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)

    trail = ws[:step + 1]
    ax.scatter(trail, [v * v for v in trail], s=14, color=MUTED, zorder=3, alpha=0.6)
    ax.scatter([w_ball], [w_ball ** 2], s=90, color=ACCENT, zorder=5,
               edgecolors=INK, linewidths=0.8)

    ax.set_title(f"第 {step} 步：w = {w_now:.4f}，损失 = {w_now ** 2:.4f}",
                 fontsize=11, color=INK)
    ax.set_xlabel("w")
    ax.set_ylabel("损失 L(w) = w²")
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-2, 20)
    ax.text(0.02, 0.95, "更新规则：w ← w − 0.1 × 2w", transform=ax.transAxes,
            fontsize=9.5, color=PRIMARY, va="top")


def plot_descent_anim():
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    total = STEPS * 2 + HOLD + 1

    def update(f):
        draw_frame(ax, f)

    anim = FuncAnimation(fig, update, frames=range(total), interval=200)
    save_gif(fig, anim, "ch05-descent-anim.gif", fps=6)


if __name__ == "__main__":
    setup_style()
    plot_descent_anim()
