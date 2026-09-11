# -*- coding: utf-8 -*-
"""生成第 18 章插图：神经元 vs 卷积核、Same 填充、最大池化。"""

from matplotlib.patches import Circle, Rectangle

from fig_common import (
    ACCENT,
    FILL,
    GRID,
    INK,
    MUTED,
    PRIMARY,
    new_figure,
    save_figure,
    setup_style,
)

BLOCK_TINTS = [FILL, "#FFEDD5", "#DCFCE7", "#FEF9C3"]


def _draw_grid(ax, x0, y0, cell, data, fills=None, fontsize=12, text_colors=None):
    """在 (x0, y0) 左下角起画一个数字网格，data 按从上到下逐行给出。"""
    rows = len(data)
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            x = x0 + j * cell
            y = y0 + (rows - 1 - i) * cell
            fc = fills[i][j] if fills else "white"
            tc = text_colors[i][j] if text_colors else INK
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=fc,
                                   edgecolor=INK, linewidth=1.0, zorder=2))
            ax.text(x + cell / 2, y + cell / 2, str(val), ha="center", va="center",
                    fontsize=fontsize, color=tc, zorder=3)


def _arrow(ax, start, end, label=None, label_offset=(0, 0.4)):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=16))
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=11, color=MUTED)


def _circle_node(ax, xy, r, text, fontsize=13):
    ax.add_patch(Circle(xy, r, facecolor=FILL, edgecolor=PRIMARY, linewidth=1.8, zorder=3))
    ax.text(*xy, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=4)


def plot_neuron_vs_kernel():
    fig, ax = new_figure(figsize=(9.6, 3.8))
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(3.9, 7.4, "第 6 章的神经元", ha="center", fontsize=12, color=INK)
    sum_xy = (4.6, 4.2)
    inputs = [("$x_1$", 1.3, 6.0, r"$\times\,w_1$"),
              ("$x_2$", 1.3, 4.2, r"$\times\,w_2$"),
              ("$x_3$", 1.3, 2.4, r"$\times\,w_3$")]
    for name, x, y, weight in inputs:
        _circle_node(ax, (x, y), 0.55, name, fontsize=12)
        _arrow(ax, (x + 0.55, y),
               (sum_xy[0] - 0.68, sum_xy[1] + (y - sum_xy[1]) * 0.3),
               weight, (0, 0.35))
    _circle_node(ax, sum_xy, 0.68, "Σ")
    _arrow(ax, (sum_xy[0], 1.5), (sum_xy[0], sum_xy[1] - 0.68), "+b", (-0.55, 0))
    _arrow(ax, (sum_xy[0] + 0.68, sum_xy[1]), (sum_xy[0] + 2.3, sum_xy[1]), "z", (0, 0.45))
    ax.text(sum_xy[0] + 2.75, sum_xy[1], "输出", ha="center", va="center",
            fontsize=10, color=MUTED)

    ax.plot([9.2, 9.2], [0.6, 7.6], color=GRID, linewidth=1.2, linestyle="--")

    ax.text(15.1, 7.4, "本章的卷积核", ha="center", fontsize=12, color=INK)
    xs = [[f"$x_{i * 3 + j + 1}$" for j in range(3)] for i in range(3)]
    ws = [[f"$w_{i * 3 + j + 1}$" for j in range(3)] for i in range(3)]
    _draw_grid(ax, 10.4, 2.5, 0.95, xs, fontsize=10)
    _draw_grid(ax, 15.4, 2.5, 0.95, ws, fontsize=10)
    ax.text(14.75, 3.93, r"$\otimes$", ha="center", va="center", fontsize=20, color=ACCENT)
    ax.text(11.83, 1.9, "输入窗口 3×3", ha="center", fontsize=10, color=MUTED)
    ax.text(16.83, 1.9, "卷积核 3×3", ha="center", fontsize=10, color=MUTED)
    ax.text(19.3, 3.93, r"$=\ \sum_i x_i \cdot w_i$", ha="center", va="center",
            fontsize=13, color=INK)
    ax.text(15.1, 0.9, "本质相同：对应相乘再求和", ha="center", fontsize=10, color=MUTED)

    save_figure(fig, "ch18-neuron-vs-kernel.svg")


def plot_padding():
    fig, ax = new_figure(figsize=(9.6, 3.0))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    data5 = [["x"] * 5 for _ in range(5)]
    fill5 = [[FILL] * 5 for _ in range(5)]
    _draw_grid(ax, 0.6, 1.3, 0.62, data5, fills=fill5, fontsize=12)
    ax.text(2.15, 4.85, "原始 5×5", ha="center", fontsize=12, color=INK)

    _arrow(ax, (4.1, 2.85), (5.5, 2.85), "四周补 0", (0, 0.45))

    data7 = [["0"] * 7 for _ in range(7)]
    fill7 = [[GRID] * 7 for _ in range(7)]
    for i in range(1, 6):
        for j in range(1, 6):
            data7[i][j] = "x"
            fill7[i][j] = FILL
    _draw_grid(ax, 5.9, 0.9, 0.56, data7, fills=fill7, fontsize=10)
    ax.text(7.86, 5.25, "补零后 7×7", ha="center", fontsize=12, color=INK)

    _arrow(ax, (10.25, 2.85), (11.65, 2.85), "3×3 卷积", (0, 0.45))

    blank5 = [[""] * 5 for _ in range(5)]
    _draw_grid(ax, 12.1, 1.3, 0.62, blank5, fills=fill5)
    ax.text(13.65, 4.85, "输出 5×5", ha="center", fontsize=12, color=INK)

    ax.text(8.2, 0.35, "Same 填充：输出尺寸与输入一致，边缘信息被保留",
            ha="center", fontsize=10, color=MUTED)

    save_figure(fig, "ch18-padding.svg")


def plot_max_pooling():
    values = [[1, 3, 2, 4],
              [5, 6, 8, 7],
              [2, 1, 3, 2],
              [9, 4, 1, 0]]
    block_max = {(0, 0): 6, (0, 1): 8, (1, 0): 9, (1, 1): 3}
    fills4 = []
    colors4 = []
    for i in range(4):
        fr, cr = [], []
        for j in range(4):
            bi, bj = i // 2, j // 2
            fr.append(BLOCK_TINTS[bi * 2 + bj])
            cr.append(ACCENT if values[i][j] == block_max[(bi, bj)] else INK)
        fills4.append(fr)
        colors4.append(cr)

    fig, ax = new_figure(figsize=(10.5, 3.2))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    _draw_grid(ax, 0.6, 1.2, 0.85, values, fills=fills4, text_colors=colors4, fontsize=13)
    ax.text(2.3, 5.05, "4×4 特征图", ha="center", fontsize=12, color=INK)

    _arrow(ax, (4.5, 2.9), (5.7, 2.9), "切成 4 个 2×2", (0, 0.5))

    cell = 0.6
    block = cell * 2
    for bi in range(2):
        for bj in range(2):
            x0 = 6.1 + bj * (block + 0.55)
            y0 = 1.35 + (1 - bi) * (block + 0.55)
            sub = [[values[bi * 2][bj * 2], values[bi * 2][bj * 2 + 1]],
                   [values[bi * 2 + 1][bj * 2], values[bi * 2 + 1][bj * 2 + 1]]]
            sub_fill = [[BLOCK_TINTS[bi * 2 + bj]] * 2 for _ in range(2)]
            _draw_grid(ax, x0, y0, cell, sub, fills=sub_fill, fontsize=10)
    ax.text(7.6, 5.05, "4 个 2×2 方块", ha="center", fontsize=12, color=INK)

    _arrow(ax, (9.75, 2.9), (10.95, 2.9), "各取最大值", (0, 0.5))

    out = [[6, 8], [9, 3]]
    out_fill = [[BLOCK_TINTS[0], BLOCK_TINTS[1]], [BLOCK_TINTS[2], BLOCK_TINTS[3]]]
    _draw_grid(ax, 11.5, 1.95, 0.95, out, fills=out_fill, fontsize=14)
    ax.text(12.45, 4.3, "输出 2×2", ha="center", fontsize=12, color=INK)

    ax.text(7.4, 0.45, "每个方块 4 个数只留最大值，数据量降为 1/4",
            ha="center", fontsize=10, color=MUTED)

    save_figure(fig, "ch18-max-pooling.svg")


if __name__ == "__main__":
    setup_style()
    plot_neuron_vs_kernel()
    plot_padding()
    plot_max_pooling()
