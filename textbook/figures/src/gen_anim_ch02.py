# -*- coding: utf-8 -*-
"""生成第 2 章动画：向量加法"首尾相接"分步绘制。"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from anim_common import save_gif
from fig_common import ACCENT, INK, MUTED, PRIMARY, setup_style

V1 = (2.0, 3.0)
V2 = (1.0, 4.0)
GROW_FRAMES = 4     # 每支箭头用几帧"长出来"
HOLD_FRAMES = 4     # 画完后的定格帧数


def draw_frame(ax, f):
    """画出第 f 帧时的全部内容（帧序号超过的阶段直接画完整）。"""
    ax.clear()
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.set_xlim(-0.5, 4.3)
    ax.set_ylim(-0.6, 8.2)
    ax.set_xticks(range(0, 5))
    ax.set_yticks(range(0, 9, 2))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    t1 = min(f / GROW_FRAMES, 1.0)                                  # [2,3] 生长进度
    t2 = min(max(0.0, (f - GROW_FRAMES) / GROW_FRAMES), 1.0)        # [1,4] 生长进度
    t3 = min(max(0.0, (f - 2 * GROW_FRAMES) / GROW_FRAMES), 1.0)    # 合向量进度

    if t1 > 0:
        ax.annotate("", xy=(V1[0] * t1, V1[1] * t1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=PRIMARY,
                                    linewidth=2.5, mutation_scale=18), zorder=3)
        ax.text(0.9, 1.05, "$[2,3]$", fontsize=12, color=PRIMARY)
    if t2 > 0:
        tip = (V1[0] + V2[0] * t2, V1[1] + V2[1] * t2)
        ax.annotate("", xy=tip, xytext=V1,
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                    linewidth=2.5, mutation_scale=18), zorder=3)
        ax.text(2.62, 5.5, "$[1,4]$", fontsize=12, color=ACCENT)
    if t3 > 0:
        ax.annotate("", xy=(V1[0] + V2[0], V1[1] + V2[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.8,
                                    linestyle=(0, (4, 3)), mutation_scale=16),
                    zorder=3)
        ax.text(3.25, 7.15, "$[3,7]$", fontsize=12, color=INK)
        ax.text(0.15, 7.7, "合向量 = 两个向量首尾相接", fontsize=10, color=INK)


def plot_vector_addition_anim():
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    total = 3 * GROW_FRAMES + HOLD_FRAMES

    def update(f):
        draw_frame(ax, f)

    anim = FuncAnimation(fig, update, frames=range(total), interval=220)
    save_gif(fig, anim, "ch02-vector-addition-anim.gif", fps=5)


if __name__ == "__main__":
    setup_style()
    plot_vector_addition_anim()
