# -*- coding: utf-8 -*-
"""教材插图统一绘图规范：字体、配色、尺寸、输出。"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
OUT_DIR = SRC_DIR.parent

INK = "#1F2937"
PRIMARY = "#2563EB"
ACCENT = "#E8590C"
MUTED = "#9CA3AF"
GRID = "#E5E7EB"
FILL = "#DBEAFE"

DEFAULT_FIGSIZE = (6.4, 3.6)


def setup_style():
    """配置全局绘图风格，所有图生成前调用一次。"""
    mpl.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.dpi": 120,
        "savefig.dpi": 120,
    })


def new_figure(figsize=DEFAULT_FIGSIZE):
    """创建按规范初始化的画布。"""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig, name):
    """保存 SVG 到 textbook/figures/ 目录。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"已生成: {path}")
    return path
