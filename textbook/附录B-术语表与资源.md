# 附录B 术语表与推荐资源

## 中英术语对照

| 中文 | 英文 | 首次出现 |
|------|------|---------|
| 函数 | function | 第1章 |
| 导数 | derivative | 第1章 |
| 链式法则 | chain rule | 第1章 |
| 向量 / 矩阵 | vector / matrix | 第2章 |
| 点积 | dot product | 第2章 |
| 矩阵乘法 | matrix multiplication | 第2章 |
| 期望值 / 方差 | expectation / variance | 第3章 |
| 正态分布 | normal distribution | 第3章 |
| 指数函数 / 对数 | exponential / logarithm | 第4章 |
| 自然常数 e | Euler's number | 第4章 |
| 偏导数 / 梯度 | partial derivative / gradient | 第5章 |
| 梯度下降 | gradient descent | 第5章 |
| 学习率 | learning rate | 第5章 |
| 神经元 | neuron | 第6章 |
| 权重 / 偏置 | weight / bias | 第6章 |
| 激活函数 | activation function | 第6章 |
| 前向传播 | forward propagation | 第7章 |
| 损失函数 | loss function | 第8章 |
| 均方误差 | MSE (mean squared error) | 第8章 |
| 交叉熵 | cross-entropy | 第8章 |
| 反向传播 | backpropagation | 第10章 |
| 随机梯度下降 | SGD (stochastic gradient descent) | 第11章 |
| 小批量 | mini-batch | 第11章 |
| 动量法 | momentum | 第11章 |
| Adam | Adam (Adaptive Moment Estimation) | 第11章 |
| Sigmoid / Tanh / ReLU | 同左 | 第12章 |
| 梯度消失 / 梯度爆炸 | vanishing / exploding gradient | 第12章 |
| Xavier 初始化 | Xavier initialization | 第12章 |
| Softmax | softmax | 第13章 |
| 归一化 / 标准化 | normalization / standardization | 第13章 |
| 过拟合 / 欠拟合 | overfitting / underfitting | 第14章 |
| 训练集 / 验证集 / 测试集 | training / validation / test set | 第14章 |
| 正则化 | regularization | 第15章 |
| L2 正则 | L2 regularization | 第15章 |
| Dropout | dropout | 第15章 |
| 早停 | early stopping | 第15章 |
| 数据增强 | data augmentation | 第16章 |
| 卷积神经网络 | CNN (convolutional neural network) | 第18章 |
| 卷积核 / 特征图 / 池化 | kernel / feature map / pooling | 第18章 |
| 循环神经网络 | RNN (recurrent neural network) | 第19章 |
| 长短期记忆网络 | LSTM (Long Short-Term Memory) | 第19章 |
| 注意力机制 | attention mechanism | 第20章 |
| Transformer | transformer | 第20章 |
| 迁移学习 | transfer learning | 第21章 |
| 预训练 / 微调 | pretraining / fine-tuning | 第21章 |
| 剪枝 / 量化 / 知识蒸馏 | pruning / quantization / knowledge distillation | 第21章 |

## 统一记号速查

| 记号 | 含义 |
|------|------|
| $x$ | 输入 |
| $y$ / $\hat{y}$ | 真实值 / 预测值 |
| $w$ | 权重 |
| $b$ | 偏置 |
| $z = wx + b$ | 加权求和（激活前） |
| $a = f(z)$ | 激活输出 |
| $L$ | 损失函数 |
| $\eta$ | 学习率 |
| $\delta$ | 误差项（反向传播中） |
| $\frac{\partial L}{\partial w}$ | 损失对权重的偏导数 |

## 推荐资源

### 视频（直觉优先）

- **3Blue1Brown《Neural Networks》系列**：动画演示神经网络、梯度下降与反向传播，
  与本书第 6~10 章一一对应，建议学完第 10 章后观看
- **StatQuest**：机器学习概念的短动画讲解，适合查缺补漏

### 书籍（由浅入深）

- Michael Nielsen《Neural Networks and Deep Learning》（免费在线）：
  用 Python 手写神经网络的 classic，可作为本书第三部分之后的进阶阅读
- Ian Goodfellow《Deep Learning》：体系完整的"花书"，适合作为工具书查阅，
  不建议从头通读
- 李沐《动手学深度学习》（免费在线）：代码驱动的深度学习教程，
  学完本书后可平滑衔接

### 课程

- 吴恩达《Machine Learning Specialization》（Coursera）：体系化的入门课
- 李沐《动手学深度学习》配套视频（B 站）：中文讲解，含 PyTorch 实践

### 练习平台

- **Kaggle Learn**：免费的微课程与实战练习
- **PyTorch / TensorFlow 官方教程**：本书毕业后，任选一个框架深入

## 配套代码

本仓库 `examples/` 下有三套由浅入深的参考实现：

- `examples/c-tutorial/`：纯 C 教学神经网络库 + 9 个训练任务，对应本书第 9、17 章的动手实验
- `examples/nn_framework/`：分层架构的工程级框架（含 GUI 训练仪表盘），是第 17 章
  "从零手写神经网络库"的进阶对照，适合想了解真实项目如何分层的读者
- `examples/painterengine-demo/`：基于 PainterEngine 的 ANN 可视化演示

除这三套工程外，全书各章的 Python 实验代码块后都内嵌了"🐘 C 语言对照"折叠块
（C99 + 标准库，与正文公式逐行对应，可独立编译运行）——它们是"一章一例"的轻量
对照；上面三套工程则是完整可玩的多任务实现，两者配合使用效果最佳。
