/**
 * trainer.h — 训练能力
 *
 * 组合 component/nn 的 forward/backward/update 形成训练能力。
 * 编排训练循环、进度回调、验证。
 */

#ifndef CAPABILITY_TRAINER_H
#define CAPABILITY_TRAINER_H

#include "component/nn/nn.h"
#include "capability/data_source.h"

typedef struct {
    int    epoch;
    int    total_epochs;
    double avg_loss;
    double elapsed_sec;
} TrainReport;

typedef void (*TrainCallback)(const TrainReport *report, void *user_data);

typedef struct {
    double final_loss;
    double elapsed_sec;
    int    correct_count;
    int    test_count;
} TrainResult;

NeuralNet* nn_trainer_build(const Task *task);
TrainResult nn_trainer_run(const Task *task, TrainCallback callback, void *user_data);
double nn_trainer_train_epoch(NeuralNet *nn, const Task *task);

#endif /* CAPABILITY_TRAINER_H */
