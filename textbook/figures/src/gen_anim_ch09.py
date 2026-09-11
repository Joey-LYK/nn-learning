# -*- coding: utf-8 -*-
"""生成第 9 章动画：1-16-1 网络拟合 sin(x) 的完整训练过程。

训练算法、随机种子与第 9 章实验 A 完全一致（纯标准库），动画在
预定的检查轮次抓取网络的预测曲线，让"看见 Loss 下降"动起来。
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from anim_common import save_gif
from fig_common import ACCENT, INK, MUTED, PRIMARY, setup_style

# ---- 第 9 章的 Net 类（与正文逐行一致，单输出简化版）----


class Net:
    def __init__(self, sizes, hidden_act="tanh", out_act="linear", lr=0.02):
        random.seed(42)
        self.lr = lr
        self.hidden_act = hidden_act
        self.out_act = out_act
        self.w = []
        self.b = []
        for i in range(1, len(sizes)):
            rows, cols = sizes[i], sizes[i - 1]
            scale = math.sqrt(2.0 / (rows + cols))
            self.w.append([[random.gauss(0, 1) * scale for _ in range(cols)]
                           for _ in range(rows)])
            self.b.append([0.0] * rows)

    def _activate(self, z, act):
        if act == "sigmoid":
            return 1.0 / (1.0 + math.exp(-z))
        if act == "tanh":
            return math.tanh(z)
        return z

    def forward(self, x):
        acts = [x]
        for i in range(len(self.w)):
            act = self.hidden_act if i < len(self.w) - 1 else self.out_act
            a_prev = acts[-1]
            a = []
            for r in range(len(self.w[i])):
                z = sum(self.w[i][r][c] * a_prev[c]
                        for c in range(len(a_prev))) + self.b[i][r]
                a.append(self._activate(z, act))
            acts.append(a)
        return acts

    def train_step(self, x, y):
        acts = self.forward(x)
        a_out = acts[-1]
        delta = [a_out[k] - y[k] for k in range(len(y))]
        if self.out_act == "sigmoid":
            delta = [d * a * (1 - a) for d, a in zip(delta, a_out)]
        for layer in range(len(self.w) - 1, -1, -1):
            a_prev = acts[layer]
            new_delta = [0.0] * len(a_prev)
            for r in range(len(self.w[layer])):
                for c in range(len(a_prev)):
                    new_delta[c] += self.w[layer][r][c] * delta[r]
                    self.w[layer][r][c] -= self.lr * delta[r] * a_prev[c]
                self.b[layer][r] -= self.lr * delta[r]
            if layer > 0:
                if self.hidden_act == "sigmoid":
                    delta = [new_delta[c] * a_prev[c] * (1 - a_prev[c])
                             for c in range(len(a_prev))]
                else:
                    delta = [new_delta[c] * (1 - a_prev[c] ** 2)
                             for c in range(len(a_prev))]

    def loss(self, data):
        total = 0.0
        for x, y in data:
            a = self.forward(x)[-1]
            total += sum((a[k] - y[k]) ** 2 for k in range(len(y)))
        return total / len(data)


# ---- 训练并抓取检查点的预测曲线 ----
CHECKPOINTS = [0, 20, 50, 100, 200, 400, 800, 1500, 2500, 4000]
GRID = np.linspace(0, 2 * math.pi, 60)

random.seed(7)
data = []
for _ in range(200):
    xv = random.uniform(0, 2 * math.pi)
    data.append(([xv], [math.sin(xv)]))

net = Net([1, 16, 1], hidden_act="tanh", out_act="linear", lr=0.02)

snapshots = []          # (epoch, loss, 预测曲线)
next_cp = 0
for epoch in range(0, CHECKPOINTS[-1] + 1):
    if next_cp < len(CHECKPOINTS) and epoch == CHECKPOINTS[next_cp]:
        preds = [net.forward([v])[-1][0] for v in GRID]
        snapshots.append((epoch, net.loss(data), preds))
        next_cp += 1
    if epoch == CHECKPOINTS[-1]:
        break
    random.shuffle(data)
    for x, y in data[:30]:
        net.train_step(x, y)

HOLD = 5      # 每个检查点定格几帧
FPS = 6


def draw_frame(ax, f):
    idx = min(f // HOLD, len(snapshots) - 1)
    epoch, loss, preds = snapshots[idx]
    ax.clear()
    ax.scatter([d[0][0] for d in data], [d[1][0] for d in data],
               s=8, color=MUTED, alpha=0.45, zorder=2,
               label="训练样本（带噪声）")
    grid_true = np.linspace(0, 2 * math.pi, 120)
    ax.plot(grid_true, np.sin(grid_true), color=INK, linewidth=1.2,
            linestyle="--", zorder=3, label="真实 sin(x)")
    ax.plot(GRID, preds, color=PRIMARY, linewidth=2.6, zorder=4,
            label="网络预测")
    ax.set_title(f"epoch {epoch:5d}   平均损失 {loss:.4f}", fontsize=11, color=INK)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 2 * math.pi)
    ax.set_ylim(-1.6, 1.6)
    ax.legend(loc="upper right", fontsize=8.5)


def plot_sin_training_anim():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    total = len(snapshots) * HOLD

    def update(f):
        draw_frame(ax, f)

    anim = FuncAnimation(fig, update, frames=range(total), interval=170)
    save_gif(fig, anim, "ch09-sin-training-anim.gif", fps=FPS)


if __name__ == "__main__":
    setup_style()
    plot_sin_training_anim()
