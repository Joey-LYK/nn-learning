# -*- coding: utf-8 -*-
"""生成第 16 章插图：归一化前后损失地形对比、特征尺度对比。"""

import matplotlib.pyplot as plt
import numpy as np

from fig_common import ACCENT, FILL, INK, MUTED, PRIMARY, new_figure, save_figure, setup_style


def plot_loss_terrain():
    grid = np.linspace(-1.6, 1.6, 200)
    w1, w2 = np.meshgrid(grid, grid)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    titles = ["不归一化：狭长山谷，震荡前进", "归一化后：接近圆形，直取谷底"]

    z_raw = 0.08 * w1 ** 2 + 2.0 * w2 ** 2
    z_norm = w1 ** 2 + w2 ** 2

    for ax, z, title in zip(axes, (z_raw, z_norm), titles):
        ax.contour(w1, w2, z, levels=6, colors=PRIMARY, linewidths=1.4)
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.scatter([0], [0], s=60, color=ACCENT, zorder=5, marker="*")

    path_x = np.array([1.4, 0.9, -0.9, 0.55, -0.5, 0.3, -0.18, 0.1, 0.0])
    path_y = np.array([1.1, -0.85, 0.8, -0.6, 0.45, -0.3, 0.18, -0.08, 0.0])
    axes[0].plot(path_x, path_y, color=ACCENT, linewidth=1.8, zorder=4,
                 marker="o", markersize=4)
    axes[1].plot([1.35, 0.6, 0.0], [1.15, 0.5, 0.0], color=ACCENT, linewidth=1.8,
                 zorder=4, marker="o", markersize=4)

    save_figure(fig, "ch16-loss-terrain.svg")


def plot_feature_scale():
    rng = np.random.default_rng(3)
    areas = rng.uniform(60, 200, 14)
    floors = rng.uniform(1, 30, 14)
    areas_n = (areas - 60) / (200 - 60)
    floors_n = (floors - 1) / (30 - 1)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.6))

    axes[0].scatter(areas, np.ones_like(areas), s=45, color=PRIMARY, zorder=4, label="面积")
    axes[0].scatter(floors, np.zeros_like(floors), s=45, color=ACCENT, zorder=4, label="楼层")
    axes[0].set_xlim(-10, 210)
    axes[0].set_title("归一化前：共用真实尺度 0~200", fontsize=12, color=INK)
    axes[0].text(100, -0.62, "楼层被挤在左端，几乎没有存在感", ha="center",
                 fontsize=10, color=MUTED)
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)

    axes[1].scatter(areas_n, np.ones_like(areas_n), s=45, color=PRIMARY, zorder=4)
    axes[1].scatter(floors_n, np.zeros_like(floors_n), s=45, color=ACCENT, zorder=4)
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_title("归一化后：共用 $[0, 1]$ 尺度", fontsize=12, color=INK)
    axes[1].text(0.5, -0.62, "两个特征占据相同宽度，权重平起平坐", ha="center",
                 fontsize=10, color=MUTED)

    for ax in axes:
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["楼层", "面积"])
        ax.set_ylim(-0.9, 1.6)

    fig.tight_layout()
    save_figure(fig, "ch16-feature-scale.svg")


if __name__ == "__main__":
    setup_style()
    plot_loss_terrain()
    plot_feature_scale()
