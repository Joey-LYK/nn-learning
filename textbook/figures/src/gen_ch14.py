# -*- coding: utf-8 -*-
"""生成第 14 章插图：拟合三形态对比、学习曲线三种形态。"""

import matplotlib.pyplot as plt
import numpy as np

from fig_common import ACCENT, INK, MUTED, PRIMARY, save_figure, setup_style

X_DATA = np.array([1, 2, 3, 4, 5], dtype=float)
Y_DATA = np.array([2.1, 3.9, 6.2, 7.8, 10.1])


def plot_fitting_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    titles = ["欠拟合\n（学不会）", "刚好\n（真学会）", "过拟合\n（背答案）"]
    notes = ["训练误差大\n新数据误差大", "训练误差小\n新数据误差小", "训练误差≈0\n新数据误差大！"]
    x_fine = np.linspace(0.5, 5.5, 300)

    for ax, title, note in zip(axes, titles, notes):
        ax.scatter(X_DATA, Y_DATA, s=45, color=INK, zorder=4)
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlim(0.3, 5.7)
        ax.set_ylim(0, 13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(note, fontsize=10, color=MUTED)

    axes[0].plot([0.5, 5.5], [5.0, 5.0], color=PRIMARY, linewidth=2.2, zorder=3)
    axes[1].plot(x_fine, 2.0 * x_fine, color=PRIMARY, linewidth=2.2, zorder=3)
    coef = np.polyfit(X_DATA, Y_DATA, 4)
    axes[2].plot(x_fine, np.polyval(coef, x_fine), color=ACCENT, linewidth=2.2, zorder=3)
    axes[2].set_ylim(-1, 13)

    save_figure(fig, "ch14-fitting-comparison.svg")


def plot_learning_curves():
    epochs = np.linspace(0, 100, 200)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    titles = ["形态一：欠拟合", "形态二：理想", "形态三：过拟合"]
    notes = ["两条都停在高位", "两条都降到低位", "训练降、验证回升"]

    train1 = 0.60 + 0.30 * np.exp(-epochs / 12.0)
    val1 = train1 + 0.06
    train2 = 0.06 + 0.95 * np.exp(-epochs / 22.0)
    val2 = 0.09 + 1.05 * np.exp(-epochs / 24.0)
    train3 = 0.04 + 0.95 * np.exp(-epochs / 26.0)
    val3 = 0.12 + 1.0 * np.exp(-epochs / 20.0) + 0.0035 * np.maximum(epochs - 35, 0) ** 1.35

    curves = [(train1, val1), (train2, val2), (train3, val3)]
    for ax, (tr, va), title, note in zip(axes, curves, titles, notes):
        ax.plot(epochs, tr, color=PRIMARY, linewidth=2.2, label="训练损失")
        ax.plot(epochs, va, color=ACCENT, linewidth=2.2, linestyle="--", label="验证损失")
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlabel("轮次\n" + note, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 50, 100])

    axes[0].set_ylabel("损失")
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)
    e_min = np.argmin(val3)
    axes[2].annotate("验证损失在此回升", xy=(epochs[e_min], val3[e_min]),
                     xytext=(48, 0.62), fontsize=10, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTED))

    save_figure(fig, "ch14-learning-curves.svg")


if __name__ == "__main__":
    setup_style()
    plot_fitting_comparison()
    plot_learning_curves()
