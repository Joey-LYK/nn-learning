/**
 * nn_reporter.h — 输出/报告服务
 *
 * 纯 I/O 层。将训练状态和验证结果格式化为人可读的输出。
 * 不知道神经网络内部状态，只接收数据。
 * 可被替换（如换成 GUI 输出）而不影响 trainer。
 */

#ifndef NN_REPORTER_H
#define NN_REPORTER_H

#include "core/nn.h"
#include "apps/demo/tasks/task.h"
#include "trainer/trainer.h"

/**
 * 打印训练进度（用于 TrainCallback）。
 */
void nn_reporter_progress(const TrainReport *report, void *user_data);

/**
 * 打印训练完成摘要。
 */
void nn_reporter_summary(const TrainResult *result);

/**
 * 打印回归验证结果。
 */
void nn_reporter_validate_regression(const Task *task, NeuralNet *nn, int count);

/**
 * 打印分类验证结果（含准确率）。
 */
void nn_reporter_validate_classification(const Task *task, NeuralNet *nn, int count);

#endif /* NN_REPORTER_H */
