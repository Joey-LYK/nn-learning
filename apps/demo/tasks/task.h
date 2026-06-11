/**
 * task.h — 任务插件接口
 *
 * 每个 task_xxx.c 文件实现一个 Task 实例，提供：
 *   1. 数据生成器（generate）
 *   2. 网络结构和训练参数的自描述（config）
 *
 * Task 是 data 层的"插件"——框架通过 Task 接口
 * 获取数据和配置，但不关心数据的具体来源。
 */

#ifndef TASK_H
#define TASK_H

#include "nn_config.h"

/* ====== 数据生成器类型 ====== */

/**
 * 生成一对 (输入, 期望输出) 训练样本。
 * @param input  输出参数，长度为 Task.input_dim
 * @param target 输出参数，长度为 Task.output_dim
 */
typedef void (*DataGenerator)(nn_float_t *input, nn_float_t *target);

/* ====== 任务网络配置 ====== */

typedef struct {
    int        hidden_count;                   /* 隐藏层数量 */
    int        hidden_sizes[NN_MAX_HIDDEN_LAYERS]; /* 各隐藏层神经元数 */
    int        hidden_act;                     /* 隐藏层激活函数（对应 core/nn.h 中的 Activation 枚举） */
    int        output_act;                     /* 输出层激活函数 */
    double     learning_rate;                  /* 推荐学习率 */
    int        epochs;                         /* 推荐训练轮次 */
    int        samples_per_epoch;             /* 每轮训练样本数 */
} TaskConfig;

/* ====== 任务描述 ====== */

typedef struct {
    const char   *name;        /* 任务名称（如 "sin(x) 函数拟合"） */
    const char   *description; /* 一句话描述（用于菜单显示） */
    int           input_dim;   /* 输入向量维度 */
    int           output_dim;  /* 输出向量维度 */
    DataGenerator generate;    /* 生成一对训练样本 */
    TaskConfig    config;      /* 网络结构和训练参数 */
} Task;

#endif /* TASK_H */
