/**
 * reporter.h — 报告能力
 *
 * 纯 I/O 层。将训练状态格式化为人可读的输出。
 */

#ifndef CAPABILITY_REPORTER_H
#define CAPABILITY_REPORTER_H

#include "capability/trainer.h"

void nn_reporter_progress(const TrainReport *report, void *user_data);
void nn_reporter_summary(const TrainResult *result);

#endif /* CAPABILITY_REPORTER_H */
