# -*- coding: utf-8 -*-
"""生成第 20 章插图：句子"猫追老鼠，它跑得很快"的注意力权重矩阵热力图。"""

import matplotlib.pyplot as plt
import numpy as np

from fig_common import (
    ACCENT,
    INK,
    new_figure,
    save_figure,
    setup_style,
)

WORDS = ["猫", "追", "老鼠", "，", "它", "跑", "得", "很", "快"]

# 与正文 ASCII 一致的定性强度：■=5 ▓=4 ▒=3 ░=2 ·=1
# "它"行让"老鼠"成为唯一的最高关注（代词指向），其余行保持对角占优
LEVELS = [
    [5, 4, 3, 1, 3, 2, 1, 1, 1],   # 猫
    [4, 5, 4, 1, 2, 4, 1, 1, 2],   # 追
    [3, 4, 5, 1, 4, 3, 1, 1, 1],   # 老鼠
    [1, 1, 1, 5, 1, 1, 1, 1, 1],   # ，
    [3, 2, 5, 1, 4, 3, 1, 1, 2],   # 它  ← 最高列是"老鼠"
    [1, 3, 2, 1, 2, 5, 4, 2, 3],   # 跑
    [1, 1, 1, 1, 1, 4, 5, 4, 3],   # 得
    [1, 1, 1, 1, 1, 2, 4, 5, 5],   # 很
    [1, 2, 1, 1, 2, 4, 3, 5, 5],   # 快
]


def plot_attention_heatmap():
    mat = np.array(LEVELS, dtype=float)
    mat = mat / mat.sum(axis=1, keepdims=True)   # 每行归一化，和为 1

    fig, ax = new_figure(figsize=(6.4, 4.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)

    for i in range(len(WORDS)):
        for j in range(len(WORDS)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 else INK)

    # 高亮"它"那一行
    row = WORDS.index("它")
    ax.add_patch(plt.Rectangle((-0.5, row - 0.5), len(WORDS), 1, fill=False,
                               edgecolor=ACCENT, linewidth=2.2))
    ax.annotate('代词"它"最关注"老鼠"', xy=(WORDS.index("老鼠"), row - 0.42),
                xytext=(3.4, -0.78), fontsize=10, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, linewidth=1.2))

    ax.set_xticks(range(len(WORDS)), WORDS, fontsize=10)
    ax.set_yticks(range(len(WORDS)), WORDS, fontsize=10)
    ax.set_xlabel("被关注的词（Key）")
    ax.set_ylabel("发出关注的词（Query）")
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "ch20-attention-heatmap.svg")


if __name__ == "__main__":
    setup_style()
    plot_attention_heatmap()
