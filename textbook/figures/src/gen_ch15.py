# -*- coding: utf-8 -*-
"""生成第 15 章插图：早停示意图。"""

import numpy as np

from fig_common import ACCENT, INK, MUTED, PRIMARY, new_figure, save_figure, setup_style


def plot_early_stopping():
    epochs = np.arange(0, 101)
    rng = np.random.default_rng(7)

    train_loss = 0.05 + 0.75 * np.exp(-epochs / 28.0)
    val_base = 0.30 + 0.55 * np.exp(-epochs / 16.0) + 0.0028 * np.maximum(epochs - 35, 0) ** 1.4
    noise = rng.normal(0, 0.006, len(epochs))
    noise[:40] = 0.0
    val_loss = val_base + noise

    best_epoch = int(np.argmin(val_loss))
    stop_epoch = min(best_epoch + 10, len(epochs) - 1)

    fig, ax = new_figure()
    ax.plot(epochs, train_loss, color=PRIMARY, linewidth=2.2, label="训练损失")
    ax.plot(epochs, val_loss, color=ACCENT, linewidth=2.2, label="验证损失")
    ax.scatter([best_epoch], [val_loss[best_epoch]], s=70, color=ACCENT, zorder=4)
    ax.axvline(best_epoch, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
    ax.axvline(stop_epoch, color=INK, linewidth=1.0, linestyle=":", zorder=2)
    ax.annotate("最低点：保存这份权重", xy=(best_epoch, val_loss[best_epoch]),
                xytext=(12, 0.62), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("连续 N 轮未创新低：停！", xy=(stop_epoch, val_loss[stop_epoch]),
                xytext=(58, 0.72), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(76, 0.34, "小波动不算", fontsize=9, color=MUTED)
    ax.set_xlabel("轮次")
    ax.set_ylabel("损失")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.9)
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "ch15-early-stopping.svg")


if __name__ == "__main__":
    setup_style()
    plot_early_stopping()
