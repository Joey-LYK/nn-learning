/**
 * nn.h — 神经网络核心 API
 *
 * 纯计算层。提供前向传播、反向传播、权重更新、模型序列化。
 * 不知道"训练循环"和"任务"的概念——这些由 engine 层负责。
 *
 * 架构概览：
 *
 *   data/task_xxx.c               core/nn.c
 *   ┌──────────────┐              ┌──────────────┐
 *   │ .generate()  │──→ 数据 ──→ │ nn_forward()  │ 前向传播
 *   │  生成训练样本  │              │ nn_backward() │ 反向传播
 *   └──────────────┘              │ nn_update()   │ 权重更新
 *                                 │ nn_train_step│ 训练一步
 *                                 └──────────────┘
 *
 * 层结构图解：
 *
 *   输入层    隐藏层1    隐藏层2    输出层
 *   [○ ○]  →  [○ ○ ○] → [○ ○ ○] →  [○]
 *              ↕ 权重      ↕ 权重     ↕ 权重
 *
 * 一个神经元的计算：
 *   net = Σ(前层输出_j × 权重_ij) + 偏置_i
 *   out = 激活函数(net)
 */

#ifndef NN_H
#define NN_H

#include "nn_config.h"

/* ========================== 神经网络层 ========================== */

typedef struct {
    int input_size;    /* 前一层神经元数量（输入层此项为 0） */
    int size;          /* 本层神经元数量 */

    double *weights;   /* 权重矩阵 [size × input_size]，行优先 */
    double *biases;    /* 偏置向量 [size] */

    double *net;       /* 加权和（激活前） [size] */
    double *out;       /* 激活后输出 [size] */
    double *error;     /* 误差项 δ [size] */

    Activation activation;
} NNLayer;

/* ========================== 神经网络 ========================== */

typedef struct {
    NNLayer *layers;
    int layer_count;
    double learning_rate;
    double total_loss;
    int total_params;
} NeuralNet;

/* ========================== 生命周期 API ========================== */

NeuralNet* nn_create(double learning_rate);
void nn_add_layer(NeuralNet *nn, int size, Activation activation);
void nn_destroy(NeuralNet *nn);

/* ========================== 推理 API ========================== */

/**
 * 前向传播。
 */
void nn_forward(NeuralNet *nn, double *input, double *output);

/**
 * 获取预测输出（等价于 nn_forward，语义更清晰）。
 */
void nn_predict(NeuralNet *nn, double *input, double *output);

/* ========================== 训练 API ========================== */

/**
 * 反向传播：计算各层误差梯度。
 * @return 损失值（MSE 或交叉熵，取决于输出层激活函数）
 */
double nn_backward(NeuralNet *nn, double *target);

/**
 * 梯度下降：用反向传播算出的梯度更新所有权重和偏置。
 */
void nn_update(NeuralNet *nn);

/**
 * 训练一步：前向 → 反向 → 更新。
 * @return 损失值
 */
double nn_train_step(NeuralNet *nn, double *input, double *target);

/* ========================== 工具 API ========================== */

void nn_print_structure(NeuralNet *nn);

/* ========================== 序列化 API ========================== */

/**
 * 导出为二进制数据。两步调用：先传 buffer=NULL 获取大小，再传 buffer 写入。
 */
int nn_export(NeuralNet *nn, void *buffer, int *size);

/**
 * 从二进制数据导入神经网络。
 */
NeuralNet* nn_import(void *buffer, int size);

#endif /* NN_H */
