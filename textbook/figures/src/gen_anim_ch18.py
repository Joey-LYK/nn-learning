# -*- coding: utf-8 -*-
"""生成第 18 章动画：3×3 卷积核在 5×5 图上滑动扫描，逐格填出特征图。

输入、卷积核与正文动手实验完全一致（中心亮点 9、拉普拉斯型核）。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from anim_common import save_gif
from fig_common import ACCENT, INK, MUTED, PRIMARY, setup_style

IMAGE = np.zeros((5, 5))
IMAGE[2, 2] = 9.0
KERNEL = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]], dtype=float)
FILL_FRAMES = 3   # 每个输出格的"高亮窗口→填入结果"各占几帧


def conv_at(i, j):
    window = IMAGE[i:i + 3, j:j + 3]
    return float((window * KERNEL).sum())


def draw_frame(ax_in, ax_out, f):
    step, phase = divmod(f, 2)
    positions = [(i, j) for i in range(3) for j in range(3)]
    done = min(step, 9)
    ax_in.clear()
    ax_out.clear()

    # 左：输入与当前窗口
    ax_in.imshow(IMAGE, cmap="Blues", vmin=-2, vmax=9)
    for r in range(5):
        for c in range(5):
            ax_in.text(c, r, f"{IMAGE[r, c]:g}", ha="center", va="center",
                       fontsize=11,
                       color="white" if IMAGE[r, c] > 4 else INK)
    if done < 9:
        i, j = positions[done]
        ax_in.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 3, 3, fill=False,
                                      edgecolor=ACCENT, linewidth=2.6))
    ax_in.set_xticks(range(5))
    ax_in.set_yticks(range(5))
    ax_in.set_title("输入 5×5（橙框 = 卷积核窗口）", fontsize=10, color=INK)
    ax_in.grid(False)

    # 右：输出特征图
    out = np.full((3, 3), np.nan)
    for k in range(done):
        i, j = positions[k]
        out[i, j] = conv_at(i, j)
    masked = np.ma.masked_invalid(out)
    ax_out.imshow(masked, cmap="Oranges", vmin=-9, vmax=72)
    for i in range(3):
        for j in range(3):
            if not np.isnan(out[i, j]):
                ax_out.text(j, i, f"{out[i, j]:g}", ha="center", va="center",
                            fontsize=12, color=INK)
    if done < 9:
        i, j = positions[done]
        v = conv_at(i, j)
        ax_out.text(j, i, f"{v:g}", ha="center", va="center", fontsize=12,
                    color=MUTED)
        ax_out.set_title(f"位置 ({i}, {j})：乘积和 = {v:g}", fontsize=10,
                         color=ACCENT)
    else:
        ax_out.set_title("特征图完成：亮点被忠实记录", fontsize=10, color=INK)
    ax_out.set_xticks(range(3))
    ax_out.set_yticks(range(3))
    ax_out.grid(False)


def plot_conv_anim():
    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    total = 9 * 2 * FILL_FRAMES + 6
    frames = range(total)

    def update(f):
        step, _ = divmod(f, FILL_FRAMES)
        draw_frame(ax_in, ax_out, step)

    anim = FuncAnimation(fig, update, frames=frames, interval=320)
    save_gif(fig, anim, "ch18-conv-anim.gif", fps=4)


if __name__ == "__main__":
    setup_style()
    plot_conv_anim()
