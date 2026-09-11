# -*- coding: utf-8 -*-
"""动画插图共享工具：把 FuncAnimation 存成 GIF（所有 Markdown 环境都支持）。"""

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from fig_common import OUT_DIR


def save_gif(fig, anim, name, fps=8):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"已生成: {path}")
    return path
