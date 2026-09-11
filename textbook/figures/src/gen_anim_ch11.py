# -*- coding: utf-8 -*-
"""生成第 11 章动画：狭长山谷里普通 GD 与动量法的赛跑（L = x² + 40y²）。

参数与第 11 章动手实验完全一致：lr = 0.01、动量 0.9、起点 (1, 1)。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from anim_common import save_gif
from fig_common import ACCENT, INK, MUTED, PRIMARY, setup_style

LR = 0.01
MU = 0.9
STEPS = 150
EVERY = 3          # 每几步取一帧
HOLD = 6


def loss(x, y):
    return x * x + 40.0 * y * y


def trajectories():
    gd = [(1.0, 1.0)]
    mom = [(1.0, 1.0)]
    vx = vy = 0.0
    xg, yg = 1.0, 1.0
    xm, ym = 1.0, 1.0
    for _ in range(STEPS):
        xg -= LR * 2 * xg
        yg -= LR * 80 * yg
        gd.append((xg, yg))

        gx, gy = 2 * xm, 80 * ym
        vx = MU * vx + gx
        vy = MU * vy + gy
        xm -= LR * vx
        ym -= LR * vy
        mom.append((xm, ym))
    return gd, mom


GD, MOM = trajectories()
FRAME_STEPS = list(range(0, STEPS + 1, EVERY))
CONTOUR_LEVELS = [0.5, 2, 8, 32, 128, 512]


def draw_frame(ax, f):
    idx = min(f // 2, len(FRAME_STEPS) - 1)
    step = FRAME_STEPS[idx]
    ax.clear()

    xs = np.linspace(-0.7, 1.25, 240)
    ys = np.linspace(-1.65, 1.25, 240)
    XX, YY = np.meshgrid(xs, ys)
    ax.contour(XX, YY, XX * XX + 40 * YY * YY, levels=CONTOUR_LEVELS,
               colors=MUTED, linewidths=0.8, zorder=1)

    gd_path = GD[:step + 1]
    mom_path = MOM[:step + 1]
    ax.plot([p[0] for p in gd_path], [p[1] for p in gd_path],
            color=INK, linewidth=1.3, zorder=3, label="普通 GD")
    ax.plot([p[0] for p in mom_path], [p[1] for p in mom_path],
            color=ACCENT, linewidth=1.8, zorder=4, label="动量法")
    ax.scatter([gd_path[-1][0]], [gd_path[-1][1]], s=45, color=INK, zorder=5)
    ax.scatter([mom_path[-1][0]], [mom_path[-1][1]], s=55, color=ACCENT,
               zorder=6, edgecolors=INK, linewidths=0.8)

    ax.set_title(f"第 {step} 步   GD 损失 {loss(*gd_path[-1]):.4f}   "
                 f"动量损失 {loss(*mom_path[-1]):.4f}", fontsize=10.5, color=INK)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.7, 1.25)
    ax.set_ylim(-1.65, 1.25)
    ax.legend(loc="upper right", fontsize=8.5)


def plot_momentum_race_anim():
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    total = len(FRAME_STEPS) * 2 + HOLD

    def update(f):
        draw_frame(ax, f)

    anim = FuncAnimation(fig, update, frames=range(total), interval=170)
    save_gif(fig, anim, "ch11-momentum-race-anim.gif", fps=7)


if __name__ == "__main__":
    setup_style()
    plot_momentum_race_anim()
