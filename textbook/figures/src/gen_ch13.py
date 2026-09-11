# -*- coding: utf-8 -*-
"""生成第 13 章插图：softmax 温度系数——同一组分数在不同 T 下的概率。"""

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

Z = [2.0, 1.0, 0.1]   # 第 4 章沿用的三组分数


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def plot_temperature():
    fig, ax = new_figure()
    ts = np.logspace(np.log10(0.28), 2, 300)
    probs = np.array([softmax([v / t for v in Z]) for t in ts])

    colors = [PRIMARY, ACCENT, INK]
    labels = ["$z_1=2.0$（分数最高）", "$z_2=1.0$", "$z_3=0.1$"]
    for j in range(3):
        ax.plot(ts, probs[:, j], color=colors[j], linewidth=2.2, label=labels[j])

    for t in (0.5, 1.0, 2.0, 100.0):
        ax.axvline(t, color=MUTED, linewidth=0.9, linestyle=":")
    for t, y in ((0.5, 1.02), (1.0, 1.02), (2.0, 1.02)):
        ax.text(t, y, f"T={t:g}", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.text(100.0, 1.02, "T=100", ha="right", va="bottom", fontsize=9, color=MUTED)

    ax.text(2.3, 0.84, "低温：尖锐\n（赢家通吃）", fontsize=10, color=INK)
    ax.text(28, 0.45, "高温：平坦\n（雨露均沾）", fontsize=10, color=INK)

    ax.set_xscale("log")
    ax.set_xlabel("温度 T（对数刻度）")
    ax.set_ylabel("各类别的概率")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(0.28, 100)
    ax.legend(loc="center left", fontsize=9, framealpha=0.9)
    save_figure(fig, "ch13-temperature.svg")


if __name__ == "__main__":
    setup_style()
    plot_temperature()
