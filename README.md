# 零基础学神经网络

一个面向**数学零基础读者**的神经网络学习项目：从数学预科讲起，配合可编译运行的示例代码，
最终落到一本系统教材。全部内容中文撰写，循序渐进，强调"手算一遍 + 动手跑一遍"。

## 仓库结构

```
├── textbook/                  # 系统教材（编写中）：零数学基础 → 神经网络算法，一章一文档
├── docs/                      # 分级学习文档
│   ├── 神经网络学习计划书.md    # 总体学习路线
│   ├── 神经网络算法学习指导文档.md
│   ├── 01-基础版-直觉与入门.md  # 直觉理解：神经元、前向传播、损失函数
│   ├── 02-进阶版-原理与推导.md  # 数学原理：反向传播推导、梯度下降
│   ├── 03-深度版-工程与实践.md  # 工程实践：优化算法、正则化、数据工程
│   ├── 04-深度进阶版-前沿架构与拓展.md  # CNN / RNN / Transformer 概览
│   └── math/                  # 数学预科：函数与导数、线性代数、概率统计等
├── examples/
│   ├── c-tutorial/            # 纯 C 教学级神经网络库 + 9 个由浅入深的训练任务
│   └── painterengine-demo/    # 基于 PainterEngine 的 ANN 可视化演示（Code::Blocks 工程）
└── third_party/
    └── PainterEngine/         # 第三方图形引擎（见 third_party/ATTRIBUTION.md）
```

## 快速开始

### 纯 C 教学示例（推荐起点）

零依赖，任意 C 编译器即可编译：

```cmd
cd examples/c-tutorial
gcc -Wall -O2 train_demo.c nn.c task_sin.c task_xor.c task_projectile.c ^
  task_comfort.c task_sevenseg.c task_temp_comp.c ^
  task_battery.c task_gesture.c task_voice.c -o train_demo.exe -lm
train_demo.exe
```

运行后通过交互菜单选择 9 个训练任务（sin 拟合、XOR、舒适度分类、电池 SOC 估计、
手势识别、语音命令识别等），详见 [examples/c-tutorial/README.md](examples/c-tutorial/README.md)。

### PainterEngine ANN 可视化演示

使用 Code::Blocks 打开 `examples/painterengine-demo/ui.cbp` 编译运行，
可交互观察神经网络训练过程与预测效果。

## 学习路线建议

1. 数学基础薄弱：先读 `docs/math/` 下的数学预科（函数与导数 → 线性代数 → 概率统计）
2. 建立直觉：读 `docs/01-基础版-直觉与入门.md`，同时跑 `examples/c-tutorial` 的基础任务
3. 理解原理：读 `docs/02-进阶版-原理与推导.md`，跟着手算反向传播
4. 工程落地：读 `docs/03-深度版-工程与实践.md`
5. 拓展视野：读 `docs/04-深度进阶版-前沿架构与拓展.md`

## 开源协议

- **代码**（`examples/` 等目录下的源码）：[MIT](LICENSE-CODE)
- **文档与教材**（`docs/`、`textbook/` 下的 Markdown）：[CC BY 4.0](LICENSE-DOCS)
- `third_party/` 下内容为第三方版权，见 [third_party/ATTRIBUTION.md](third_party/ATTRIBUTION.md)

## 贡献

欢迎提交 Issue 与 Pull Request。教材仍在编写中，如发现有误的内容或更好的讲法，欢迎指出。
