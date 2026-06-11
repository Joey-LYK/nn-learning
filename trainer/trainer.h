/**
 * nn_trainer.h — 训练引擎
 *
 * 编排层。从 Task 的 config 构建网络，执行训练循环，
 * 通过回调上报进度。不关心具体的 Task 是什么。
 *
 * 调用链：app → nn_trainer_run() → task->generate() → core/nn.c
 */

#ifndef NN_TRAINER_H
#define NN_TRAINER_H

#include "core/nn.h"
#include "apps/demo/tasks/task.h"

/* ====== 训练报告 ====== */

typedef struct {
    int    epoch;
    int    total_epochs;
    double avg_loss;
    double elapsed_sec;
} TrainReport;

/* ====== 训练回调 ====== */

/**
 * 每轮训练结束后的回调函数。
 * 由 app 层注册，用于驱动 reporter 服务。
 */
typedef void (*TrainCallback)(const TrainReport *report, void *user_data);

/* ====== 训练结果 ====== */

typedef struct {
    double final_loss;       /* 训练结束后的最终 loss */
    double elapsed_sec;       /* 训练耗时（秒） */
    int    correct_count;    /* 分类验证正确数（回归任务为 -1） */
    int    test_count;        /* 验证样本总数 */
} TrainResult;

/* ====== API ====== */

/**
 * 从 Task 的 config 构建神经网络。
 * @return 已构建的网络，调用者负责 nn_destroy()
 */
NeuralNet* nn_trainer_build(const Task *task);

/**
 * 执行完整训练流程。
 * 构建网络 → 训练循环 → 验证 → 销毁网络。
 *
 * @param task     数据源和配置
 * @param callback 进度回调（传 NULL 则不报告）
 * @param user_data 回调用户数据
 * @return 训练结果
 */
TrainResult nn_trainer_run(const Task *task, TrainCallback callback, void *user_data);

#endif /* NN_TRAINER_H */
